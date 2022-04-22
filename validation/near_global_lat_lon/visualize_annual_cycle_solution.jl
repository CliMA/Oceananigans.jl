using Statistics
using JLD2
using Printf
using GLMakie
using Oceananigans
using Oceananigans.Units
using Oceananigans.Fields: fill_halo_regions!, ReducedField

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using CUDA: @allowscalar

#####
##### Grid
#####

latitude = (-84.375, 84.375)
Δφ = latitude[2] - latitude[1]

# 2.8125 degree resolution
Nx = 128
Ny = 60
Nz = 18

using DataDeps

path = "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/lat_lon_bathymetry_and_fluxes/"

dh = DataDep("near_global_lat_lon",
    "Forcing data for global latitude longitude simulation", path * "bathymetry_lat_lon_128x60_FP32.bin"
)

DataDeps.register(dh)

datadep"near_global_lat_lon"

bathymetry_data = Array{Float32}(undef, Nx*Ny)
bathymetry_path = @datadep_str "near_global_lat_lon/bathymetry_lat_lon_128x60_FP32.bin"
read!(bathymetry_path, bathymetry_data)

bathymetry_data = bswap.(bathymetry_data) |> Array{Float64}
bathymetry = reshape(bathymetry_data, Nx, Ny)

Nmonths = 12
bytes = sizeof(Float32) * Nx * Ny

H = 3600.0
#bathymetry = - H .* (bathymetry .< -10)

# A spherical domain
underlying_grid = LatitudeLongitudeGrid(size = (Nx, Ny, Nz),
                                        longitude = (-180, 180),
                                        latitude = latitude,
                                        halo = (3, 3, 3),
                                        z = (-H, 0))

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

function geographic2cartesian(λ, φ; r=1)
    λ = cat(λ, λ[1:1] .+ 360, dims=1)

    Nλ = length(λ)
    Nφ = length(φ)

    λ = repeat(reshape(λ, Nλ, 1), 1, Nφ) 
    φ = repeat(reshape(φ, 1, Nφ), Nλ, 1)

    λ_azimuthal = λ .+ 180  # Convert to λ ∈ [0°, 360°]
    φ_azimuthal = 90 .- φ   # Convert to φ ∈ [0°, 180°] (0° at north pole)

    x = @. r * cosd(λ_azimuthal) * sind(φ_azimuthal)
    y = @. r * sind(λ_azimuthal) * sind(φ_azimuthal)
    z = @. r * cosd(φ_azimuthal)

    return x, y, z
end

λu, ϕu, ru = nodes((Face, Center, Center), grid)
λv, ϕv, rv = nodes((Center, Face, Center), grid)
λc, ϕc, rc = nodes((Center, Center, Center), grid)

xu, yu, zu = geographic2cartesian(λu, ϕu)
xv, yv, zv = geographic2cartesian(λv, ϕv)
xc, yc, zc = geographic2cartesian(λc, ϕc)

output_prefix = "annual_cycle_global_lat_lon_128_60_18_temp"

surface_file = jldopen(output_prefix * "_surface.jld2")
bottom_file = jldopen(output_prefix * "_bottom.jld2")

iterations = parse.(Int, keys(surface_file["timeseries/t"]))

iter = Node(0)

# Continents
land_ccc = bathymetry .> - underlying_grid.Δzᵃᵃᶜ / 2

function mask_land_cfc(v)
    land_cfc = cat(land_ccc, land_ccc[:, end:end], dims=2)
    v[land_cfc] .= NaN
    return v
end

function mask_land(data)
    data[land_ccc] .= NaN
    return data
end

ηi(iter) = mask_land(surface_file["timeseries/η/" * string(iter)][:, :, 1])
ui(iter) = mask_land(surface_file["timeseries/u/" * string(iter)][:, :, 1])
vi(iter) = mask_land_cfc(surface_file["timeseries/v/" * string(iter)][:, :, 1])
Ti(iter) = mask_land(surface_file["timeseries/T/" * string(iter)][:, :, 1])
ti(iter) = prettytime(surface_file["timeseries/t/" * string(iter)])

ubi(iter) = mask_land(bottom_file["timeseries/u/" * string(iter)][:, :, 1])
vbi(iter) = mask_land_cfc(bottom_file["timeseries/v/" * string(iter)][:, :, 1])

uri = ReducedField(Face, Center, Nothing, CPU(), grid, dims=3)
vri = ReducedField(Center, Face, Nothing, CPU(), grid, dims=3)
spi = ReducedField(Center, Center, Nothing, CPU(), grid, dims=3)

bathymetry = cat(bathymetry, bathymetry[1:1, :], dims=1)
land = bathymetry .> -10
water = @. !(land)

continents = 1.0 .* land
continents[water] .= NaN

sp = @lift begin
    uri .= ui($iter)
    vri .= vi($iter)
    fill_halo_regions!(uri)
    fill_halo_regions!(vri)
    spi .= sqrt(uri^2 + vri^2)
    fill_halo_regions!(spi)
    sp = spi[1:Nx+1, 1:Ny, 1]
    # sp[land] .= NaN
    return sp
end

η = @lift ηi($iter) 
u = @lift ui($iter)
v = @lift vi($iter)
T = @lift Ti($iter)

T_sphere = @lift begin
    T = cat(Ti($iter), Ti($iter)[1:1, :], dims=1)
    T[land] .= NaN
    return T
end

v_sphere = @lift begin
    v = cat(vi($iter)[:, 1:Ny], vi($iter)[1:1, 1:Ny], dims=1)
    v[land] .= NaN
    return v
end

u_sphere = @lift begin
    u = cat(ui($iter), ui($iter)[1:1, :], dims=1)
    u[land] .= NaN
    return u
end

ub = @lift ubi($iter)
vb = @lift vbi($iter)

max_η = 2
min_η = - max_η
max_u = 0.1
min_u = - max_u
max_T = 32
min_T = 0

dλbg = 0.01
dφbg = 0.01
λbg = range(-180, stop=180 - dλbg, step=dλbg)
φbg = range(-89.5, stop=89.5 - dφbg, step=dφbg)
xbg, ybg, zbg = geographic2cartesian(λbg, φbg)

sphere_continents!(ax_T) = surface!(ax_T, xbg, ybg, zbg, color=fill("#080", 1, 1))

#####
##### Sphere
#####

sphere_fig = Figure(resolution = (150, 90))

graylim = 1.0
α = 0.99

#surface!(ax, xc, yc, zc, color=sp, colormap=:blues, colorrange=(0, max_u),
#         show_axis=false, shading=false, ssao=true)

# Temperature
ax_T = sphere_fig[1, 1:3] = LScene(sphere_fig)

surface!(ax_T, xc, yc, zc, color=T_sphere, colormap=:thermal, colorrange=(0, max_T),
         show_axis=false, shading=false, ssao=true)

sphere_continents!(ax_T)

rotate_cam!(ax_T.scene, (π/8, π/6, 0))

# Meridional velocity
ax_v = sphere_fig[1, 4:6] = LScene(sphere_fig)

surface!(ax_v, xc, yc, zc, color=u_sphere, colormap=:balance, colorrange=(min_u, max_u),
         show_axis=false, shading=false, ssao=true)

sphere_continents!(ax_v)

rotate_cam!(ax_v.scene, (π/8, π/6, 0))

plot_title = @lift "t = " * ti($iter)
supertitle = sphere_fig[0, 3:4] = Label(sphere_fig, plot_title, textsize=40)

#v_title = sphere_fig[0, 5] = Label(sphere_fig, "Ocean meridional velocity", textsize=20)
#T_title = sphere_fig[0, 2] = Label(sphere_fig, "Ocean temperature", textsize=20)

display(sphere_fig)

save_interval = 5days
full_rotation_savepoints = round(Int, 10year / save_interval) 
dθ = 2π / full_rotation_savepoints
dϕ = π/4 / full_rotation_savepoints

function dϕi(i)
    savepoint = i/iterations[end] * length(iterations)
    mod_savepoint = mod1(savepoint, full_rotation_savepoints)
    halfway_savepoint = 1/2 * length(iterations)
    return mod_i < halfway_savepoint ? dϕ : - dϕ
end

GLMakie.record(sphere_fig, output_prefix * "_sphere.mp4", iterations, framerate=8) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    iter[] = i
    rotate_cam!(ax_T.scene, cameracontrols(ax_T.scene), (dϕi(i), dθ, 0))
    rotate_cam!(ax_v.scene, cameracontrols(ax_v.scene), (dϕi(i), dθ, 0))
end

#=
#####
##### Meridional velocity
#####

plane_fig = Figure(resolution = (1200, 900))

# kwargs = NamedTuple()
kwargs = (; interpolate=true)

function continents!(ax)
    Nλ = length(λc)
    Nφ = length(ϕc)
    λ = repeat(reshape(λc, Nλ, 1), 1, Nφ) 
    φ = repeat(reshape(ϕc, 1, Nφ), Nλ, 1)
    surface!(ax, λ, φ, zeros(Nx, Ny) .- 0.02, color=fill("#6e7f80", 1, 1))
    return nothing
end

ax = Axis(plane_fig[1, 1], title="Free surface displacement (m)")
continents!(ax)
hm = heatmap!(ax, λc, ϕc, η; colorrange=(min_η, max_η), colormap=:balance, kwargs...)
cb = Colorbar(plane_fig[1, 2], hm)

ax = Axis(plane_fig[2, 1], title="Sea surface temperature (ᵒC)")
continents!(ax)
hm = heatmap!(ax, λc, ϕc, T; colorrange=(min_T, max_T), colormap=:thermal, kwargs...)
cb = Colorbar(plane_fig[2, 2], hm)

ax = Axis(plane_fig[1, 3], title="East-west surface velocity (m s⁻¹)")
continents!(ax)
hm = heatmap!(ax, λc, ϕc, u; colorrange=(min_u, max_u), colormap=:balance, kwargs...)
cb = Colorbar(plane_fig[1, 4], hm)

ax = Axis(plane_fig[2, 3], title="North-south surface velocity (m s⁻¹)")
continents!(ax)
hm = heatmap!(ax, λc, ϕv, v; colorrange=(min_u, max_u), colormap=:balance, kwargs...)
cb = Colorbar(plane_fig[2, 4], hm)

ax = Axis(plane_fig[3, 1], title="East-west bottom velocity (m s⁻¹)")
continents!(ax)
hm = heatmap!(ax, λc, ϕc, ub; colorrange=(min_u, max_u), colormap=:balance, kwargs...)
cb = Colorbar(plane_fig[3, 2], hm)

ax = Axis(plane_fig[3, 3], title="North-south bottom velocity (m s⁻¹)")
continents!(ax)
hm = heatmap!(ax, λc, ϕv, vb; colorrange=(min_u, max_u), colormap=:balance, kwargs...)
cb = Colorbar(plane_fig[3, 4], hm)

title_str = @lift "Earth day = " * ti($iter)
ax_t = plane_fig[0, :] = Label(plane_fig, title_str)

GLMakie.record(plane_fig, output_prefix * "_plane.mp4", iterations, framerate=8) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    iter[] = i
end

display(plane_fig)
=#

close(surface_file)
close(bottom_file)
