using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.MultiRegion: fill_cubed_sphere_halo_regions!
#=
Install Imaginocean.jl from GitHub:
using Pkg; Pkg.add(url="https://github.com/navidcy/Imaginocean.jl", rev="main")
=#
using OffsetArrays, CairoMakie, Imaginocean, JLD2

# First create a conformal cubed sphere grid.

Nx = 30
Ny = 30
Nz = 1

radius = 1

grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz), z = (-1, 0), radius)

#=
Let's create a field. We choose a field that lives on the center of the cells. We set the field values to something and
see how that looks.
=#

field = CenterField(grid)

set!(field, (λ, φ, z) -> (sind(3λ) + 1/3 * sind(5λ)) * cosd(3φ)^2)

#=
2D visualization

We can visualize this field in 2D using a heatmap. Imaginocean.jl has a method called `heatlatlon!` that plots a field
that lives on a grid whose native coordinates are latitude and longitude.
=#

kwargs = (colorrange = (-1, 1), colormap = :balance)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "longitude [ᵒ]", ylabel = "latitude [ᵒ]", limits = ((-180, 180), (-90, 90)))
heatlatlon!(ax, field, 1; kwargs...)
save("multi_region_cubed_sphere_c0_heatlatlon.png", fig)

#=
We can do the same but with a `GeoAxis` provided by the GeoMakie.jl package that allows us to easily add coastlines or
also use various projections.
=#

using GeoMakie

fig = Figure()
ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
heatlatlon!(ax, field, 1; kwargs...)
#=
save("multi_region_cubed_sphere_c0_geo_heatlatlon.png", fig)
=#

#=
3D visualization on the sphere

To make a 3D visualization on the sphere we first create a 3D axis and then use `heatsphere!` method from
Imaginocean.jl.
=#

fig = Figure()
ax = Axis3(fig[1, 1], aspect = (1, 1, 1), limits = ((-1, 1), (-1, 1), (-1, 1)))
heatsphere!(ax, field; kwargs...)
hidedecorations!(ax) # hides the axes labels
save("multi_region_cubed_sphere_c0_heatsphere.png", fig)

c = CenterField(grid)
set!(c, (λ, φ, z) -> φ)
colorrange = (-90, 90)
colormap = :balance

for _ in 1:3
    fill_halo_regions!(c)
end

fig = Figure()
ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
heatsphere!(ax, c, 1; colorrange, colormap)
save("multi_region_cubed_sphere_c_heatsphere.png", fig)

fig = Figure()
ax = Axis(fig[1, 1])
heatlatlon!(ax, c, 1; colorrange, colormap)
save("multi_region_cubed_sphere_c_heatlatlon.png", fig)

fig = Figure(resolution = (1200, 600))
ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
heatlatlon!(ax, c, 1; colorrange, colormap)
#=
save("multi_region_cubed_sphere_c_geo_heatlatlon.png", fig)
=#

u = XFaceField(grid)
set!(u, (λ, φ, z) -> φ)

v = YFaceField(grid)
set!(v, (λ, φ, z) -> φ)

fill_cubed_sphere_halo_regions!((u, v), (Face(), Center()), (Center(), Face()))

fig = Figure()
ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
heatsphere!(ax, u, 1; colorrange, colormap)
save("multi_region_cubed_sphere_u_heatsphere.png", fig)

fig = Figure()
ax = Axis(fig[1, 1])
heatlatlon!(ax, u, 1; colorrange, colormap)
save("multi_region_cubed_sphere_u_heatlatlon.png", fig)

#=
fig = Figure(resolution = (1200, 600))
ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
heatlatlon!(ax, u, 1; colorrange, colormap)
save("multi_region_cubed_sphere_u_geo_heatlatlon.png", fig)
=#

fig = Figure()
ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
heatsphere!(ax, v, 1; colorrange, colormap)
save("multi_region_cubed_sphere_v_heatsphere.png", fig)

fig = Figure()
ax = Axis(fig[1, 1])
heatlatlon!(ax, v, 1; colorrange, colormap)
save("multi_region_cubed_sphere_v_heatlatlon.png", fig)

#=
fig = Figure(resolution = (1200, 600))
ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
heatlatlon!(ax, v, 1; colorrange, colormap)
save("multi_region_cubed_sphere_v_geo_heatlatlon.png", fig)
=#

# Comparison of coordinates and metrics of a 32x32 cubed sphere grid with 4 halos relative to their counterparts from MITgcm

Nx, Ny, Nz = 32, 32, 1
cs_grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz), z = (-1, 0), radius=6370e3, horizontal_direction_halo = 4,
                                     z_halo = 1)
Hx, Hy, Hz = cs_grid.Hx, cs_grid.Hy, cs_grid.Hz

using DataDeps
cs32_4 = DataDep("cubed_sphere_32_grid_with_4_halos",
                 "Conformal cubed sphere grid with 32×32 cells on each face and 4 halos on each side",
                 "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cs32_with_4_halos/cubed_sphere_32_grid_with_4_halos.jld2",
                 "fbe684cb560c95ecae627b23784e449aa083a1e6e029dcda32cbfecfc0e26721")
DataDeps.register(cs32_4)
grid_filepath = datadep"cubed_sphere_32_grid_with_4_halos/cubed_sphere_32_grid_with_4_halos.jld2"
cs_grid_MITgcm = ConformalCubedSphereGrid(grid_filepath;
                                          Nz = 1,
                                          z = (-1, 0),
                                          panel_halo = (4, 4, 1),
                                          radius = 6370e3)

vars = (:λᶜᶜᵃ, :λᶠᶠᵃ, :φᶜᶜᵃ, :φᶠᶠᵃ, :Δxᶜᶜᵃ, :Δxᶠᶜᵃ, :Δxᶜᶠᵃ, :Δxᶠᶠᵃ, :Δyᶜᶜᵃ, :Δyᶠᶜᵃ, :Δyᶜᶠᵃ, :Δyᶠᶠᵃ, :Azᶜᶜᵃ, :Azᶠᶜᵃ,
        :Azᶜᶠᵃ, :Azᶠᶠᵃ)

var_diffs = Tuple(Symbol(string(var) * "_difference_MITgcm") for var in vars)

for var_diff in var_diffs
    eval(:($var_diff = zeros(Nx+2Hx, Ny+2Hy, 6)))
end

jldopen("cs_grid_difference_with_MITgcm.jld2", "w") do file
    for panel in 1:6
        for var_diff in var_diffs
            var_diff_name = string(var_diff)
            expr = quote
                $file[$var_diff_name * "/" * string($panel)] = $var_diff[:, :, $panel]
            end
            eval(expr)
        end
    end
end
