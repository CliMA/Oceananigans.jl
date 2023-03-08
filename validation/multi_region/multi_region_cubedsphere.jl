using Oceananigans

using Oceananigans.MultiRegion: getregion
using Oceananigans.Utils: Iterate,
                          get_lat_lon_nodes_and_vertices,
                          get_cartesian_nodes_and_vertices
using Oceananigans.BoundaryConditions: fill_halo_regions!

using GeoMakie, GLMakie
GLMakie.activate!()


Nx, Ny = 32, 32

grid = ConformalCubedSphereGrid(panel_size=(Nx, Ny, 1), z=(-1, 0), radius=1.0)

c = CenterField(grid)

regions = Iterate(Tuple(j for j in 1:length(grid.partition)))

set!(c, regions)

colorrange = (1, 6)
colormap = :Accent_6

#=
@apply_regionally set!(c, (x, y, z) -> cosd(3x)^2 * sind(3y))
colorrange = (-1, 1)
colormap = :balance
=#

fill_halo_regions!(c)


fig = Figure()
ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))

for region in [1, 2, 3, 4, 5, 6]
    (xc, yc, zc), (xvertices, yvertices, zvertices) = get_cartesian_nodes_and_vertices(Center(), Center(), getregion(c.grid, region))

    quad_points3 = vcat([Point3{eltype(grid)}.(xvertices[:, i, j], yvertices[:, i, j], zvertices[:, i, j]) for i in 1:size(xvertices, 2), j in 1:size(xvertices, 3)]...)
    quad_faces = vcat([begin; j = (i-1) * 4 + 1; [j j+1  j+2; j+2 j+3 j]; end for i in 1:length(quad_points3)÷4]...)

    colors_per_point = vcat(fill.(vec(getregion(c, region)), 4)...)

    mesh!(ax, quad_points3, quad_faces; color = colors_per_point, shading = false, colorrange, colormap)
    cl = lines!(ax, GeoMakie.coastlines(), color = :black, linewidth=0.85)

    translate!(cl, 0, 0, 1000)
end

fig


fig = Figure()
ax = Axis(fig[1, 1])

for region in [1, 2, 3, 4, 5, 6]
    (λc, φc), (λvertices, φvertices) = get_lat_lon_nodes_and_vertices(Center(), Center(), getregion(c.grid, region))

    quad_points = vcat([Point2{eltype(grid)}.(λvertices[:, i, j], φvertices[:, i, j]) for i in 1:size(λvertices, 2), j in 1:size(λvertices, 3)]...)
    quad_faces = vcat([begin; j = (i-1) * 4 + 1; [j j+1  j+2; j+2 j+3 j]; end for i in 1:length(quad_points)÷4]...)

    colors_per_point = vcat(fill.(vec(getregion(c, region)), 4)...)

    mesh!(ax, quad_points, quad_faces; color = colors_per_point, shading = false, colorrange, colormap)
end
xlims!(ax, (-180, 180))
ylims!(ax, (-90, 90))

fig


fig = Figure(resolution = (1200, 600))

ax = GeoAxis(fig[1, 1],
             coastlines = true,
             lonlims = automatic)

for region in [1, 2, 3, 4, 5, 6]
    (λc, φc), (λvertices, φvertices) = get_lat_lon_nodes_and_vertices(Center(), Center(), getregion(c.grid, region))

    quad_points = vcat([Point2{eltype(grid)}.(λvertices[:, i, j], φvertices[:, i, j]) for i in 1:size(λvertices, 2), j in 1:size(λvertices, 3)]...)
    quad_faces = vcat([begin; j = (i-1) * 4 + 1; [j j+1  j+2; j+2 j+3 j]; end for i in 1:length(quad_points)÷4]...)

    colors_per_point = vcat(fill.(vec(getregion(c, region)), 4)...)

    mesh!(ax, quad_points, quad_faces; color = colors_per_point, shading = false, colorrange, colormap)
    cl = lines!(ax, GeoMakie.coastlines(), color = :black, linewidth=0.85)

    translate!(cl, 0, 0, 1000)
end

fig
