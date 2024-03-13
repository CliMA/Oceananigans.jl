using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: fill_paired_halo_regions!
#=
Install Imaginocean.jl from GitHub:
using Pkg; Pkg.add(url="https://github.com/navidcy/Imaginocean.jl", rev="main")
=#
using CairoMakie, Imaginocean, JLD2

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
save("multi_region_cubed_sphere_c0_geo_heatlatlon.png", fig)

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
save("multi_region_cubed_sphere_c_geo_heatlatlon.png", fig)

u = XFaceField(grid)
set!(u, (λ, φ, z) -> φ)

v = YFaceField(grid)
set!(v, (λ, φ, z) -> φ)

fill_paired_halo_regions!((u, v))

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

Nx, Ny, Nz = 4, 4, 1
grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz), z = (-1, 0), radius=1, horizontal_direction_halo = 2,
                                  z_halo = 1)
Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

jldopen("cubed-sphere-dynamics_branch_cs_grid.jld2", "w") do file
    for region in 1:6
        file["λᶜᶜᵃ/" * string(region)]  =  grid[region].λᶜᶜᵃ
        file["λᶠᶜᵃ/" * string(region)]  =  grid[region].λᶠᶜᵃ
        file["λᶜᶠᵃ/" * string(region)]  =  grid[region].λᶜᶠᵃ
        file["λᶠᶠᵃ/" * string(region)]  =  grid[region].λᶠᶠᵃ
        file["φᶜᶜᵃ/" * string(region)]  =  grid[region].φᶜᶜᵃ
        file["φᶠᶜᵃ/" * string(region)]  =  grid[region].φᶠᶜᵃ
        file["φᶜᶠᵃ/" * string(region)]  =  grid[region].φᶜᶠᵃ
        file["φᶠᶠᵃ/" * string(region)]  =  grid[region].φᶠᶠᵃ
        file["Δxᶜᶜᵃ/" * string(region)] = grid[region].Δxᶜᶜᵃ
        file["Δxᶠᶜᵃ/" * string(region)] = grid[region].Δxᶠᶜᵃ
        file["Δxᶜᶠᵃ/" * string(region)] = grid[region].Δxᶜᶠᵃ
        file["Δxᶠᶠᵃ/" * string(region)] = grid[region].Δxᶠᶠᵃ
        file["Δyᶜᶜᵃ/" * string(region)] = grid[region].Δyᶜᶜᵃ
        file["Δyᶠᶜᵃ/" * string(region)] = grid[region].Δyᶠᶜᵃ
        file["Δyᶜᶠᵃ/" * string(region)] = grid[region].Δyᶜᶠᵃ
        file["Δyᶠᶠᵃ/" * string(region)] = grid[region].Δyᶠᶠᵃ
        file["Azᶜᶜᵃ/" * string(region)] = grid[region].Azᶜᶜᵃ
        file["Azᶠᶜᵃ/" * string(region)] = grid[region].Azᶠᶜᵃ
        file["Azᶜᶠᵃ/" * string(region)] = grid[region].Azᶜᶠᵃ
        file["Azᶠᶠᵃ/" * string(region)] = grid[region].Azᶠᶠᵃ
    end
end
