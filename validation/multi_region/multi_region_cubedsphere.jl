using Oceananigans

using Oceananigans.MultiRegion: getregion
using Oceananigans.Utils: Iterate,
                          get_lat_lon_nodes_and_vertices,
                          get_cartesian_nodes_and_vertices
using Oceananigans.BoundaryConditions: fill_halo_regions!

using GeoMakie, GLMakie
GLMakie.activate!()

function heatsphere!(ax::Axis3, field::Field, k=1; kwargs...)
    LX, LY, LZ = location(field)
    grid = field.grid

    _, (xvertices, yvertices, zvertices) = get_cartesian_nodes_and_vertices(grid, LX(), LY(), LZ())

    quad_points3 = vcat([Point3.(xvertices[:, i, j], yvertices[:, i, j], zvertices[:, i, j]) for i in axes(xvertices, 2), j in axes(xvertices, 3)]...)
    quad_faces = vcat([begin; j = (i-1) * 4 + 1; [j j+1  j+2; j+2 j+3 j]; end for i in 1:length(quad_points3)÷4]...)
    
    colors_per_point = vcat(fill.(vec(interior(field, :, :, k)), 4)...)
    
    mesh!(ax, quad_points3, quad_faces; color = colors_per_point, shading = false, kwargs...)

    return ax
end

function heatlatlon!(ax::Axis, field::Field, k=1; kwargs...)
    LX, LY, LZ = location(field)
    grid = field.grid

    _, (λvertices, φvertices) = get_lat_lon_nodes_and_vertices(grid, LX(), LY(), LZ())

    quad_points = vcat([Point2.(λvertices[:, i, j], φvertices[:, i, j]) for i in axes(λvertices, 2), j in axes(λvertices, 3)]...)
    quad_faces = vcat([begin; j = (i-1) * 4 + 1; [j j+1  j+2; j+2 j+3 j]; end for i in 1:length(quad_points)÷4]...)

    colors_per_point = vcat(fill.(vec(interior(field, :, :, k)), 4)...)

    mesh!(ax, quad_points, quad_faces; color = colors_per_point, shading = false, kwargs...)

    xlims!(ax, (-180, 180))
    ylims!(ax, (-90, 90))

    return ax
end



Nx, Ny, Nz = 16, 16, 2

grid = ConformalCubedSphereGrid(panel_size=(Nx, Ny, Nz), z=(-1, 0), radius=1, panel_halo = (3, 3, 1), panel_topology=(Bounded, Bounded, Bounded))

c = CenterField(grid)

regions = Iterate(Tuple(j for j in 1:length(grid.partition)))

set!(c, regions)

colorrange = (1, 6)
colormap = :Accent_6


@apply_regionally set!(c, (x, y, z) -> cosd(3x)^2 * sind(3y))
colorrange = (-1, 1)

@apply_regionally set!(c, (x, y, z) -> y)
colorrange = (-90, 90)
colormap = :balance

colorrange = (1, Ny)
for region in 1:6, j in 1:Ny, i in 1:Nx
    getregion(c, region).data[i, j, 1] = j
end


fill_halo_regions!(c)


fig = Figure()
ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))

for region in [1, 2, 3, 4, 5, 6]
    heatsphere!(ax, getregion(c, region); colorrange, colormap)
end

fig


fig = Figure()
ax = Axis(fig[1, 1])

for region in [1, 2, 3, 4, 5, 6]
    heatlatlon!(ax, getregion(c, region); colorrange, colormap)
end

fig


fig = Figure(resolution = (1200, 600))

ax = GeoAxis(fig[1, 1],
             coastlines = true,
             lonlims = automatic)

for region in [1, 2, 3, 4, 5, 6]
    heatlatlon!(ax, getregion(c, region); colorrange, colormap)
end

fig
