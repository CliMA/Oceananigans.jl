using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: fill_cubed_sphere_halo_regions!
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
                 "https://github.com/CliMA/OceananigansArtifacts.jl/raw/ncc-sb/add-cs32-grid-with-4-halos/cubed_sphere_grids/cs32_with_4_halos/cubed_sphere_32_grid_with_4_halos.jld2",
                 "356b4ec09dbf9817e96ee6b6f1d6ec3acd53e689bc105e436497898830145e2a")
DataDeps.register(cs32_4)
grid_filepath = datadep"cubed_sphere_32_grid_with_4_halos/cubed_sphere_32_grid_with_4_halos.jld2"
cs_grid_MITgcm = ConformalCubedSphereGrid(grid_filepath;
                                          Nz = 1,
                                          z = (-1, 0),
                                          panel_halo = (4, 4, 1),
                                          radius = 6370e3)

λᶜᶜᵃ_difference_MITgcm  = OffsetArray(zeros(Nx+2Hx, Ny+2Hy, 6), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:6)
λᶠᶠᵃ_difference_MITgcm  = OffsetArray(zeros(Nx+2Hx, Ny+2Hy, 6), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:6)
φᶜᶜᵃ_difference_MITgcm  = OffsetArray(zeros(Nx+2Hx, Ny+2Hy, 6), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:6)
φᶠᶠᵃ_difference_MITgcm  = OffsetArray(zeros(Nx+2Hx, Ny+2Hy, 6), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:6)
Δxᶜᶜᵃ_difference_MITgcm = OffsetArray(zeros(Nx+2Hx, Ny+2Hy, 6), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:6)
Δxᶠᶜᵃ_difference_MITgcm = OffsetArray(zeros(Nx+2Hx, Ny+2Hy, 6), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:6)
Δxᶜᶠᵃ_difference_MITgcm = OffsetArray(zeros(Nx+2Hx, Ny+2Hy, 6), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:6)
Δxᶠᶠᵃ_difference_MITgcm = OffsetArray(zeros(Nx+2Hx, Ny+2Hy, 6), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:6)
Δyᶜᶜᵃ_difference_MITgcm = OffsetArray(zeros(Nx+2Hx, Ny+2Hy, 6), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:6)
Δyᶠᶜᵃ_difference_MITgcm = OffsetArray(zeros(Nx+2Hx, Ny+2Hy, 6), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:6)
Δyᶜᶠᵃ_difference_MITgcm = OffsetArray(zeros(Nx+2Hx, Ny+2Hy, 6), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:6)
Δyᶠᶠᵃ_difference_MITgcm = OffsetArray(zeros(Nx+2Hx, Ny+2Hy, 6), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:6)
Azᶜᶜᵃ_difference_MITgcm = OffsetArray(zeros(Nx+2Hx, Ny+2Hy, 6), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:6)
Azᶠᶜᵃ_difference_MITgcm = OffsetArray(zeros(Nx+2Hx, Ny+2Hy, 6), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:6)
Azᶜᶠᵃ_difference_MITgcm = OffsetArray(zeros(Nx+2Hx, Ny+2Hy, 6), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:6)
Azᶠᶠᵃ_difference_MITgcm = OffsetArray(zeros(Nx+2Hx, Ny+2Hy, 6), 1-Hx:Nx+Hx, 1-Hy:Ny+Hy, 1:6)

jldopen("cs_grid_difference_with_MITgcm.jld2", "w") do file
    for region in 1:6
        #=
        λᶜᶜᵃ_difference_MITgcm[:, :, region]  =  cs_grid[region].λᶜᶜᵃ -  OffsetArray(cs_grid_MITgcm[region].λᶜᶜᵃ[1:end-Hx, 1:end-Hy], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        λᶠᶠᵃ_difference_MITgcm[:, :, region]  =  cs_grid[region].λᶠᶠᵃ -  OffsetArray(cs_grid_MITgcm[region].λᶠᶠᵃ[1:end-Hx, 1:end-Hy], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        φᶜᶜᵃ_difference_MITgcm[:, :, region]  =  cs_grid[region].φᶜᶜᵃ -  OffsetArray(cs_grid_MITgcm[region].φᶜᶜᵃ[1:end-Hx, 1:end-Hy], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        φᶠᶠᵃ_difference_MITgcm[:, :, region]  =  cs_grid[region].φᶠᶠᵃ -  OffsetArray(cs_grid_MITgcm[region].φᶠᶠᵃ[1:end-Hx, 1:end-Hy], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Δxᶜᶜᵃ_difference_MITgcm[:, :, region] = cs_grid[region].Δxᶜᶜᵃ - OffsetArray(cs_grid_MITgcm[region].Δxᶜᶜᵃ[1:end-Hx, 1:end-Hy], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Δxᶠᶜᵃ_difference_MITgcm[:, :, region] = cs_grid[region].Δxᶠᶜᵃ - OffsetArray(cs_grid_MITgcm[region].Δxᶠᶜᵃ[1:end-Hx, 1:end-Hy], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Δxᶜᶠᵃ_difference_MITgcm[:, :, region] = cs_grid[region].Δxᶜᶠᵃ - OffsetArray(cs_grid_MITgcm[region].Δxᶜᶠᵃ[1:end-Hx, 1:end-Hy], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Δxᶠᶠᵃ_difference_MITgcm[:, :, region] = cs_grid[region].Δxᶠᶠᵃ - OffsetArray(cs_grid_MITgcm[region].Δxᶠᶠᵃ[1:end-Hx, 1:end-Hy], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Δyᶜᶜᵃ_difference_MITgcm[:, :, region] = cs_grid[region].Δyᶜᶜᵃ - OffsetArray(cs_grid_MITgcm[region].Δyᶜᶜᵃ[1:end-Hx, 1:end-Hy], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Δyᶠᶜᵃ_difference_MITgcm[:, :, region] = cs_grid[region].Δyᶠᶜᵃ - OffsetArray(cs_grid_MITgcm[region].Δyᶠᶜᵃ[1:end-Hx, 1:end-Hy], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Δyᶜᶠᵃ_difference_MITgcm[:, :, region] = cs_grid[region].Δyᶜᶠᵃ - OffsetArray(cs_grid_MITgcm[region].Δyᶜᶠᵃ[1:end-Hx, 1:end-Hy], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Δyᶠᶠᵃ_difference_MITgcm[:, :, region] = cs_grid[region].Δyᶠᶠᵃ - OffsetArray(cs_grid_MITgcm[region].Δyᶠᶠᵃ[1:end-Hx, 1:end-Hy], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Azᶜᶜᵃ_difference_MITgcm[:, :, region] = cs_grid[region].Azᶜᶜᵃ - OffsetArray(cs_grid_MITgcm[region].Azᶜᶜᵃ[1:end-Hx, 1:end-Hy], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Azᶠᶜᵃ_difference_MITgcm[:, :, region] = cs_grid[region].Azᶠᶜᵃ - OffsetArray(cs_grid_MITgcm[region].Azᶠᶜᵃ[1:end-Hx, 1:end-Hy], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Azᶜᶠᵃ_difference_MITgcm[:, :, region] = cs_grid[region].Azᶜᶠᵃ - OffsetArray(cs_grid_MITgcm[region].Azᶜᶠᵃ[1:end-Hx, 1:end-Hy], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Azᶠᶠᵃ_difference_MITgcm[:, :, region] = cs_grid[region].Azᶠᶠᵃ - OffsetArray(cs_grid_MITgcm[region].Azᶠᶠᵃ[1:end-Hx, 1:end-Hy], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        =#
        file["λᶜᶜᵃ_difference_MITgcm/" * string(region)]  =  λᶜᶜᵃ_difference_MITgcm[:, :, region]
        file["λᶠᶠᵃ_difference_MITgcm/" * string(region)]  =  λᶠᶠᵃ_difference_MITgcm[:, :, region]
        file["φᶜᶜᵃ_difference_MITgcm/" * string(region)]  =  φᶜᶜᵃ_difference_MITgcm[:, :, region]
        file["φᶠᶠᵃ_difference_MITgcm/" * string(region)]  =  φᶠᶠᵃ_difference_MITgcm[:, :, region]
        file["Δxᶜᶜᵃ_difference_MITgcm/" * string(region)] = Δxᶜᶜᵃ_difference_MITgcm[:, :, region]
        file["Δxᶠᶜᵃ_difference_MITgcm/" * string(region)] = Δxᶠᶜᵃ_difference_MITgcm[:, :, region]
        file["Δxᶜᶠᵃ_difference_MITgcm/" * string(region)] = Δxᶜᶠᵃ_difference_MITgcm[:, :, region]
        file["Δxᶠᶠᵃ_difference_MITgcm/" * string(region)] = Δxᶠᶠᵃ_difference_MITgcm[:, :, region]
        file["Δyᶜᶜᵃ_difference_MITgcm/" * string(region)] = Δyᶜᶜᵃ_difference_MITgcm[:, :, region]
        file["Δyᶠᶜᵃ_difference_MITgcm/" * string(region)] = Δyᶠᶜᵃ_difference_MITgcm[:, :, region]
        file["Δyᶜᶠᵃ_difference_MITgcm/" * string(region)] = Δyᶜᶠᵃ_difference_MITgcm[:, :, region]
        file["Δyᶠᶠᵃ_difference_MITgcm/" * string(region)] = Δyᶠᶠᵃ_difference_MITgcm[:, :, region]
        file["Azᶜᶜᵃ_difference_MITgcm/" * string(region)] = Azᶜᶜᵃ_difference_MITgcm[:, :, region]
        file["Azᶠᶜᵃ_difference_MITgcm/" * string(region)] = Azᶠᶜᵃ_difference_MITgcm[:, :, region]
        file["Azᶜᶠᵃ_difference_MITgcm/" * string(region)] = Azᶜᶠᵃ_difference_MITgcm[:, :, region]
        file["Azᶠᶠᵃ_difference_MITgcm/" * string(region)] = Azᶠᶠᵃ_difference_MITgcm[:, :, region]
    end
end
