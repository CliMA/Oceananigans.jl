using Oceananigans

using Oceananigans.Fields: replace_horizontal_vector_halos!
using Oceananigans.Utils: apply_regionally!
using Oceananigans.BoundaryConditions: fill_halo_regions!

# install Imaginocean.jl from GitHub
# using Pkg; Pkg.add(url="https://github.com/navidcy/Imaginocean.jl", rev="main")
using Imaginocean

using GLMakie
Makie.inline!(false)
GLMakie.activate!()

using GeoMakie

#=
using Oceananigans.Utils: get_lat_lon_nodes_and_vertices, 

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

heatlatlon!(ax::Axis, field::CubedSphereField, k=1; kwargs...) =
    apply_regionally!(heatlatlon!, ax, field, k; kwargs...)
=#

Nx, Ny, Nz = 4, 4, 1
grid = ConformalCubedSphereGrid(panel_size=(Nx, Ny, Nz), z=(-1, 0), radius=1, horizontal_direction_halo = 3, 
                                z_topology=Bounded)

c = CenterField(grid)

set!(c, (λ, φ, z) -> φ)
colorrange = (-90, 90)
colormap = :balance

fill_halo_regions!(c)

fig = Figure()
ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
heatsphere!(ax, c; colorrange, colormap)
fig

save("multi_region_cubed_sphere_c_heatsphere.png", fig)

fig = Figure()
ax = Axis(fig[1, 1])
heatlatlon!(ax, c; colorrange, colormap)
fig

save("multi_region_cubed_sphere_c_heatlatlon.png", fig)

fig = Figure(size=(1200, 600))
ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
heatlatlon!(ax, c; colorrange, colormap)
fig

save("multi_region_cubed_sphere_c_geo_latlon.png", fig)

u = XFaceField(grid)
set!(u, (λ, φ, z) -> φ)

v = YFaceField(grid)
set!(v, (λ, φ, z) -> φ)

for _ in 1:2
    fill_halo_regions!(u)
    fill_halo_regions!(v)
    @apply_regionally replace_horizontal_vector_halos!((; u, v, w = nothing), grid)
end

fig = Figure()
ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
heatsphere!(ax, u; colorrange, colormap)
fig

save("multi_region_cubed_sphere_u_heatsphere.png", fig)

fig = Figure()
ax = Axis(fig[1, 1])
heatlatlon!(ax, u; colorrange, colormap)
fig
save("multi_region_cubed_sphere_u_heatlatlon.png", fig)

# fig = Figure(size=(1200, 600))
# ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
# heatlatlon!(ax, u; colorrange, colormap)
# fig
# save("multi_region_cubed_sphere_u_geo_latlon.png", fig)

fig = Figure()
ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
heatsphere!(ax, v; colorrange, colormap)
fig

save("multi_region_cubed_sphere_v_heatsphere.png", fig)

fig = Figure()
ax = Axis(fig[1, 1])
heatlatlon!(ax, v; colorrange, colormap)
fig

save("multi_region_cubed_sphere_v_heatlatlon.png", fig)

# fig = Figure(size=(1200, 600))
# ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
# heatlatlon!(ax, v; colorrange, colormap)
# fig

# save("multi_region_cubed_sphere_v_geo_latlon.png", fig)   
