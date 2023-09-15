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

function multi_region_cubed_sphere_plots()

    Nx, Ny, Nz = 5, 5, 2
    grid = ConformalCubedSphereGrid(panel_size=(Nx, Ny, Nz), z=(-1, 0), radius=1, horizontal_direction_halo = 3, 
                                    z_topology=Bounded)
    
    c = CenterField(grid)
    
    set!(c, (x, y, z) -> y)
    colorrange = (-90, 90)
    colormap = :balance
    
    fill_halo_regions!(c)
    
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
    heatsphere!(ax, c; colorrange, colormap)
    save("multi_region_cubed_sphere_c_heatsphere.png", fig)
    
    fig = Figure()
    ax = Axis(fig[1, 1])
    heatlatlon!(ax, c; colorrange, colormap)
    save("multi_region_cubed_sphere_c_heatlatlon.png", fig)

    fig = Figure(resolution = (1200, 600))
    ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
    heatlatlon!(ax, c; colorrange, colormap)
    save("multi_region_cubed_sphere_c_geo_latlon.png", fig)
    
    u = XFaceField(grid)
    set!(u, (x, y, z) -> y)

    v = YFaceField(grid)
    set!(v, (x, y, z) -> y)   
    
    for _ in 1:2
        fill_halo_regions!(u)
        fill_halo_regions!(v)
        @apply_regionally replace_horizontal_vector_halos!((; u = u, v = v, w = nothing), grid)
    end
    
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
    heatsphere!(ax, u; colorrange, colormap)
    save("multi_region_cubed_sphere_u_heatsphere.png", fig)
    
    fig = Figure()
    ax = Axis(fig[1, 1])
    heatlatlon!(ax, u; colorrange, colormap)
    save("multi_region_cubed_sphere_u_heatlatlon.png", fig)

    fig = Figure(resolution = (1200, 600))
    ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
    heatlatlon!(ax, u; colorrange, colormap)
    save("multi_region_cubed_sphere_u_geo_latlon.png", fig)
    
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
    heatsphere!(ax, v; colorrange, colormap)
    save("multi_region_cubed_sphere_v_heatsphere.png", fig)
    
    fig = Figure()
    ax = Axis(fig[1, 1])
    heatlatlon!(ax, v; colorrange, colormap)
    save("multi_region_cubed_sphere_v_heatlatlon.png", fig)

    fig = Figure(resolution = (1200, 600))
    ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
    heatlatlon!(ax, v; colorrange, colormap)
    save("multi_region_cubed_sphere_v_geo_latlon.png", fig)   
    
end

test_multi_region_cubed_sphere_plots = true
if test_multi_region_cubed_sphere_plots
    multi_region_cubed_sphere_plots()
end