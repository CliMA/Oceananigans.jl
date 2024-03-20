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

# Comparison of 4x4 cubed sphere grid coordinates and metrics relative to their counterparts from the
# cubed-sphere-dynamics branch adopting a two-pass halo-filling approach

Nx, Ny, Nz = 4, 4, 1
grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz), z = (-1, 0), radius=1, horizontal_direction_halo = 2,
                                  z_halo = 1)
Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

jldopen("cs-grid-metrics_branch_cs_grid.jld2", "w") do file
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

cubed_sphere_dynamics_branch_cs_grid_file = jldopen("cubed-sphere-dynamics_branch_cs_grid.jld2")

λᶜᶜᵃ_difference  = zeros(Nx+2Hx, Ny+2Hy, 6)
λᶠᶜᵃ_difference  = zeros(Nx+2Hx, Ny+2Hy, 6)
λᶜᶠᵃ_difference  = zeros(Nx+2Hx, Ny+2Hy, 6)
λᶠᶠᵃ_difference  = zeros(Nx+2Hx, Ny+2Hy, 6)
φᶜᶜᵃ_difference  = zeros(Nx+2Hx, Ny+2Hy, 6)
φᶠᶜᵃ_difference  = zeros(Nx+2Hx, Ny+2Hy, 6)
φᶜᶠᵃ_difference  = zeros(Nx+2Hx, Ny+2Hy, 6)
φᶠᶠᵃ_difference  = zeros(Nx+2Hx, Ny+2Hy, 6)
Δxᶜᶜᵃ_difference = zeros(Nx+2Hx, Ny+2Hy, 6)
Δxᶠᶜᵃ_difference = zeros(Nx+2Hx, Ny+2Hy, 6)
Δxᶜᶠᵃ_difference = zeros(Nx+2Hx, Ny+2Hy, 6)
Δxᶠᶠᵃ_difference = zeros(Nx+2Hx, Ny+2Hy, 6)
Δyᶜᶜᵃ_difference = zeros(Nx+2Hx, Ny+2Hy, 6)
Δyᶠᶜᵃ_difference = zeros(Nx+2Hx, Ny+2Hy, 6)
Δyᶜᶠᵃ_difference = zeros(Nx+2Hx, Ny+2Hy, 6)
Δyᶠᶠᵃ_difference = zeros(Nx+2Hx, Ny+2Hy, 6)
Azᶜᶜᵃ_difference = zeros(Nx+2Hx, Ny+2Hy, 6)
Azᶠᶜᵃ_difference = zeros(Nx+2Hx, Ny+2Hy, 6)
Azᶜᶠᵃ_difference = zeros(Nx+2Hx, Ny+2Hy, 6)
Azᶠᶠᵃ_difference = zeros(Nx+2Hx, Ny+2Hy, 6)

jldopen("cs_grid_difference.jld2", "w") do file
    for region in 1:6
        λᶜᶜᵃ_difference[:, :, region]  =  grid[region].λᶜᶜᵃ - cubed_sphere_dynamics_branch_cs_grid_file["λᶜᶜᵃ/"  * string(region)]
        λᶠᶜᵃ_difference[:, :, region]  =  grid[region].λᶠᶜᵃ - cubed_sphere_dynamics_branch_cs_grid_file["λᶠᶜᵃ/"  * string(region)]
        λᶜᶠᵃ_difference[:, :, region]  =  grid[region].λᶜᶠᵃ - cubed_sphere_dynamics_branch_cs_grid_file["λᶜᶠᵃ/"  * string(region)]
        λᶠᶠᵃ_difference[:, :, region]  =  grid[region].λᶠᶠᵃ - cubed_sphere_dynamics_branch_cs_grid_file["λᶠᶠᵃ/"  * string(region)]
        φᶜᶜᵃ_difference[:, :, region]  =  grid[region].φᶜᶜᵃ - cubed_sphere_dynamics_branch_cs_grid_file["φᶜᶜᵃ/"  * string(region)]
        φᶠᶜᵃ_difference[:, :, region]  =  grid[region].φᶠᶜᵃ - cubed_sphere_dynamics_branch_cs_grid_file["φᶠᶜᵃ/"  * string(region)]
        φᶜᶠᵃ_difference[:, :, region]  =  grid[region].φᶜᶠᵃ - cubed_sphere_dynamics_branch_cs_grid_file["φᶜᶠᵃ/"  * string(region)]
        φᶠᶠᵃ_difference[:, :, region]  =  grid[region].φᶠᶠᵃ - cubed_sphere_dynamics_branch_cs_grid_file["φᶠᶠᵃ/"  * string(region)]
        Δxᶜᶜᵃ_difference[:, :, region] = grid[region].Δxᶜᶜᵃ - cubed_sphere_dynamics_branch_cs_grid_file["Δxᶜᶜᵃ/" * string(region)]
        Δxᶠᶜᵃ_difference[:, :, region] = grid[region].Δxᶠᶜᵃ - cubed_sphere_dynamics_branch_cs_grid_file["Δxᶠᶜᵃ/" * string(region)]
        Δxᶜᶠᵃ_difference[:, :, region] = grid[region].Δxᶜᶠᵃ - cubed_sphere_dynamics_branch_cs_grid_file["Δxᶜᶠᵃ/" * string(region)]
        Δxᶠᶠᵃ_difference[:, :, region] = grid[region].Δxᶠᶠᵃ - cubed_sphere_dynamics_branch_cs_grid_file["Δxᶠᶠᵃ/" * string(region)]
        Δyᶜᶜᵃ_difference[:, :, region] = grid[region].Δyᶜᶜᵃ - cubed_sphere_dynamics_branch_cs_grid_file["Δyᶜᶜᵃ/" * string(region)]
        Δyᶠᶜᵃ_difference[:, :, region] = grid[region].Δyᶠᶜᵃ - cubed_sphere_dynamics_branch_cs_grid_file["Δyᶠᶜᵃ/" * string(region)]
        Δyᶜᶠᵃ_difference[:, :, region] = grid[region].Δyᶜᶠᵃ - cubed_sphere_dynamics_branch_cs_grid_file["Δyᶜᶠᵃ/" * string(region)]
        Δyᶠᶠᵃ_difference[:, :, region] = grid[region].Δyᶠᶠᵃ - cubed_sphere_dynamics_branch_cs_grid_file["Δyᶠᶠᵃ/" * string(region)]
        Azᶜᶜᵃ_difference[:, :, region] = grid[region].Azᶜᶜᵃ - cubed_sphere_dynamics_branch_cs_grid_file["Azᶜᶜᵃ/" * string(region)]
        Azᶠᶜᵃ_difference[:, :, region] = grid[region].Azᶠᶜᵃ - cubed_sphere_dynamics_branch_cs_grid_file["Azᶠᶜᵃ/" * string(region)]
        Azᶜᶠᵃ_difference[:, :, region] = grid[region].Azᶜᶠᵃ - cubed_sphere_dynamics_branch_cs_grid_file["Azᶜᶠᵃ/" * string(region)]
        Azᶠᶠᵃ_difference[:, :, region] = grid[region].Azᶠᶠᵃ - cubed_sphere_dynamics_branch_cs_grid_file["Azᶠᶠᵃ/" * string(region)]
        file["λᶜᶜᵃ_difference/" * string(region)]  =  λᶜᶜᵃ_difference[:, :, region]
        file["λᶠᶜᵃ_difference/" * string(region)]  =  λᶠᶜᵃ_difference[:, :, region]
        file["λᶜᶠᵃ_difference/" * string(region)]  =  λᶜᶠᵃ_difference[:, :, region]
        file["λᶠᶠᵃ_difference/" * string(region)]  =  λᶠᶠᵃ_difference[:, :, region]
        file["φᶜᶜᵃ_difference/" * string(region)]  =  φᶜᶜᵃ_difference[:, :, region]
        file["φᶠᶜᵃ_difference/" * string(region)]  =  φᶠᶜᵃ_difference[:, :, region]
        file["φᶜᶠᵃ_difference/" * string(region)]  =  φᶜᶠᵃ_difference[:, :, region]
        file["φᶠᶠᵃ_difference/" * string(region)]  =  φᶠᶠᵃ_difference[:, :, region]
        file["Δxᶜᶜᵃ_difference/" * string(region)] = Δxᶜᶜᵃ_difference[:, :, region]
        file["Δxᶠᶜᵃ_difference/" * string(region)] = Δxᶠᶜᵃ_difference[:, :, region]
        file["Δxᶜᶠᵃ_difference/" * string(region)] = Δxᶜᶠᵃ_difference[:, :, region]
        file["Δxᶠᶠᵃ_difference/" * string(region)] = Δxᶠᶠᵃ_difference[:, :, region]
        file["Δyᶜᶜᵃ_difference/" * string(region)] = Δyᶜᶜᵃ_difference[:, :, region]
        file["Δyᶠᶜᵃ_difference/" * string(region)] = Δyᶠᶜᵃ_difference[:, :, region]
        file["Δyᶜᶠᵃ_difference/" * string(region)] = Δyᶜᶠᵃ_difference[:, :, region]
        file["Δyᶠᶠᵃ_difference/" * string(region)] = Δyᶠᶠᵃ_difference[:, :, region]
        file["Azᶜᶜᵃ_difference/" * string(region)] = Azᶜᶜᵃ_difference[:, :, region]
        file["Azᶠᶜᵃ_difference/" * string(region)] = Azᶠᶜᵃ_difference[:, :, region]
        file["Azᶜᶠᵃ_difference/" * string(region)] = Azᶜᶠᵃ_difference[:, :, region]
        file["Azᶠᶠᵃ_difference/" * string(region)] = Azᶠᶠᵃ_difference[:, :, region]
    end
end

close(cubed_sphere_dynamics_branch_cs_grid_file)

# Comparison of 32x32 cubed sphere grid coordinates and metrics relative to their counterparts from MITgcm

Nx, Ny, Nz = 32, 32, 1
grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz), z = (-1, 0), radius=6370e3, horizontal_direction_halo = 4,
                                  z_halo = 1)
Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

MITgcm_cs_grid_file = jldopen("jmc_cubed_sphere_32_grid_with_4_halos.jld2")

λᶜᶜᵃ_difference_MITgcm  = zeros(Nx+2Hx, Ny+2Hy, 6)
λᶠᶠᵃ_difference_MITgcm  = zeros(Nx+2Hx, Ny+2Hy, 6)
φᶜᶜᵃ_difference_MITgcm  = zeros(Nx+2Hx, Ny+2Hy, 6)
φᶠᶠᵃ_difference_MITgcm  = zeros(Nx+2Hx, Ny+2Hy, 6)
Δxᶠᶜᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)
Δxᶜᶠᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)
Δyᶠᶜᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)
Δyᶜᶠᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)
Δyᶠᶠᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)
Azᶜᶜᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)
Azᶠᶜᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)
Azᶜᶠᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)
Azᶠᶠᵃ_difference_MITgcm = zeros(Nx+2Hx, Ny+2Hy, 6)

jldopen("MITgcm_cs_grid_difference.jld2", "w") do file
    for region in 1:6
        λᶜᶜᵃ_difference_MITgcm[:, :, region]  =  grid[region].λᶜᶜᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/λᶜᶜᵃ" ], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        λᶠᶠᵃ_difference_MITgcm[:, :, region]  =  grid[region].λᶠᶠᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/λᶠᶠᵃ" ], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        φᶜᶜᵃ_difference_MITgcm[:, :, region]  =  grid[region].φᶜᶜᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/φᶜᶜᵃ" ], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        φᶠᶠᵃ_difference_MITgcm[:, :, region]  =  grid[region].φᶠᶠᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/φᶠᶠᵃ" ], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Δxᶠᶜᵃ_difference_MITgcm[:, :, region] = grid[region].Δxᶠᶜᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Δxᶠᶜᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Δxᶜᶠᵃ_difference_MITgcm[:, :, region] = grid[region].Δxᶜᶠᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Δxᶜᶠᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Δyᶠᶜᵃ_difference_MITgcm[:, :, region] = grid[region].Δyᶠᶜᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Δyᶠᶜᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Δyᶜᶠᵃ_difference_MITgcm[:, :, region] = grid[region].Δyᶜᶠᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Δyᶜᶠᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Azᶜᶜᵃ_difference_MITgcm[:, :, region] = grid[region].Azᶜᶜᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Azᶜᶜᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Azᶠᶜᵃ_difference_MITgcm[:, :, region] = grid[region].Azᶠᶜᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Azᶠᶜᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Azᶜᶠᵃ_difference_MITgcm[:, :, region] = grid[region].Azᶜᶠᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Azᶜᶠᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        Azᶠᶠᵃ_difference_MITgcm[:, :, region] = grid[region].Azᶠᶠᵃ - OffsetArray(MITgcm_cs_grid_file["face" * string(region) * "/Azᶠᶠᵃ"], 1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
        file["λᶜᶜᵃ_difference_MITgcm/" * string(region)]  =  λᶜᶜᵃ_difference_MITgcm[:, :, region]
        file["λᶠᶠᵃ_difference_MITgcm/" * string(region)]  =  λᶠᶠᵃ_difference_MITgcm[:, :, region]
        file["φᶜᶜᵃ_difference_MITgcm/" * string(region)]  =  φᶜᶜᵃ_difference_MITgcm[:, :, region]
        file["φᶠᶠᵃ_difference_MITgcm/" * string(region)]  =  φᶠᶠᵃ_difference_MITgcm[:, :, region]
        file["Δxᶠᶜᵃ_difference_MITgcm/" * string(region)] = Δxᶠᶜᵃ_difference_MITgcm[:, :, region]
        file["Δxᶜᶠᵃ_difference_MITgcm/" * string(region)] = Δxᶜᶠᵃ_difference_MITgcm[:, :, region]
        file["Δyᶠᶜᵃ_difference_MITgcm/" * string(region)] = Δyᶠᶜᵃ_difference_MITgcm[:, :, region]
        file["Δyᶜᶠᵃ_difference_MITgcm/" * string(region)] = Δyᶜᶠᵃ_difference_MITgcm[:, :, region]
        file["Azᶜᶜᵃ_difference_MITgcm/" * string(region)] = Azᶜᶜᵃ_difference_MITgcm[:, :, region]
        file["Azᶠᶜᵃ_difference_MITgcm/" * string(region)] = Azᶠᶜᵃ_difference_MITgcm[:, :, region]
        file["Azᶜᶠᵃ_difference_MITgcm/" * string(region)] = Azᶜᶠᵃ_difference_MITgcm[:, :, region]
        file["Azᶠᶠᵃ_difference_MITgcm/" * string(region)] = Azᶠᶠᵃ_difference_MITgcm[:, :, region]
    end
end

close(MITgcm_cs_grid_file)
