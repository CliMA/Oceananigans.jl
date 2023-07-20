using Oceananigans

using Oceananigans.Architectures: architecture
using Oceananigans.Grids: halo_size
using Oceananigans.MultiRegion: getregion
using Oceananigans.Utils: Iterate, get_lat_lon_nodes_and_vertices, get_cartesian_nodes_and_vertices, apply_regionally!
using Oceananigans.BoundaryConditions: fill_halo_regions!

using GLMakie
Makie.inline!(false)
GLMakie.activate!()

using GeoMakie

function recreate_with_bounded_panels(grid::ConformalCubedSphereGrid)

    arch, FT = architecture(grid), eltype(grid)
    Nx, Ny, Nz = size(grid)
    
    horizontal_direction_halo, _, z_halo = halo_size(grid)

    z = (getregion(grid, 1).zᵃᵃᶠ[1], getregion(grid, 1).zᵃᵃᶠ[grid.Nz+1])

    radius = getregion(grid, 1).radius

    partition = grid.partition

    return ConformalCubedSphereGrid(arch, FT;
                                    panel_size = (Nx, Ny, Nz),
                                    z, horizontal_direction_halo, z_halo,
                                    radius,
                                    partition,
                                    horizontal_topology = Bounded)
end

function heatsphere!(ax::Axis3, field, k=1; kwargs...)

    LX, LY, LZ = location(field)

    grid = field.grid
    _, (xvertices, yvertices, zvertices) = get_cartesian_nodes_and_vertices(grid, LX(), LY(), LZ())

    quad_points3 = vcat([Point3.(xvertices[:, i, j], yvertices[:, i, j], zvertices[:, i, j]) 
                        for i in axes(xvertices, 2), j in axes(xvertices, 3)]...)
    quad_faces = vcat([begin; j = (i-1) * 4 + 1; [j j+1  j+2; j+2 j+3 j]; end for i in 1:length(quad_points3)÷4]...)

    colors_per_point = vcat(fill.(vec(interior(field, :, :, k)), 4)...)

    mesh!(ax, quad_points3, quad_faces; color = colors_per_point, shading = false, kwargs...)
    return ax
end

function heatlatlon!(ax::Axis, field, k=1; kwargs...)

    LX, LY, LZ = location(field)

    grid = field.grid
    _, (λvertices, φvertices) = get_lat_lon_nodes_and_vertices(grid, LX(), LY(), LZ())

    quad_points = vcat([Point2.(λvertices[:, i, j], φvertices[:, i, j]) 
                        for i in axes(λvertices, 2), j in axes(λvertices, 3)]...)
    quad_faces = vcat([begin; j = (i-1) * 4 + 1; [j j+1  j+2; j+2 j+3 j]; end for i in 1:length(quad_points)÷4]...)

    colors_per_point = vcat(fill.(vec(interior(field, :, :, k)), 4)...)

    mesh!(ax, quad_points, quad_faces; color = colors_per_point, shading = false, kwargs...)

    xlims!(ax, (-180, 180))
    ylims!(ax, (-90, 90))

    return ax
end

heatlatlon!(ax::Axis, field::CubedSphereField, k=1; kwargs...)  = apply_regionally!(heatlatlon!, ax, field, k; kwargs...)
heatsphere!(ax::Axis3, field::CubedSphereField, k=1; kwargs...) = apply_regionally!(heatsphere!, ax, field, k; kwargs...)

Nx, Ny, Nz = 10, 10, 2
grid = ConformalCubedSphereGrid(panel_size=(Nx, Ny, Nz), z=(-1, 0), radius=1, horizontal_direction_halo = 3, 
                                z_topology=Bounded)

c = CenterField(grid)
cᵢ(λ, φ, ζ) = cosd(4λ) * sind(3φ)

set!(c, cᵢ)

∂c∂φ_numerical = Field(∂y(c))
compute!(∂c∂φ_numerical)

∂c∂φ_analytic = YFaceField(grid)

∂c∂φᵢ(λ, φ, ζ) = - 3 * cosd(4λ) * cosd(3φ)
set!(∂c∂φ_analytic, ∂c∂φᵢ)

colorrange = (-1, 1)
colormap = :balance

colorrange = (-3, 3)

fig = Figure()
ax1 = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
ax2 = Axis3(fig[1, 2], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
ax3 = Axis3(fig[1, 3], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))

heatsphere!(ax1, ∂c∂φ_analytic; colorrange, colormap)
heatsphere!(ax2, ∂c∂φ_numerical; colorrange, colormap)
# heatsphere!(ax3, abs.(∂c∂φ_numerical - ∂c∂φ_analytic); colorrange, colormap)
fig
