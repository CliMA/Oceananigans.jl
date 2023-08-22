using Oceananigans

using Oceananigans.Architectures: architecture
using Oceananigans.Grids: halo_size, φnodes, λnodes
using Oceananigans.MultiRegion: getregion
using Oceananigans.Utils: Iterate, get_lat_lon_nodes_and_vertices, get_cartesian_nodes_and_vertices, apply_regionally!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_velocity_halos!

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

function panel_wise_visualization(profile, field, field_name, k; # Here k represents the vertical index of the field.
                                  hide_decorations = false, extrema_reduction_factor = 0.75)

    # Create figure.
    fig = Figure(resolution = (1350, 900))
    colorrange = (minimum(field) * extrema_reduction_factor, maximum(field) * extrema_reduction_factor)
    colormap = :solar
    
    # Plot panel 1.
    ax = Axis(fig[3, 1])
    heatmap!(ax, getregion(field, 1).data.parent[:, :, k]; colorrange, colormap)
    if hide_decorations
        hidedecorations!(ax)
    end
    
    # Plot panel 2.
    ax = Axis(fig[3, 2])
    heatmap!(ax, getregion(field, 2).data.parent[:, :, k]; colorrange, colormap)
    if hide_decorations
        hidedecorations!(ax)
    end
    
    # Plot panel 3.
    ax = Axis(fig[2, 2])
    heatmap!(ax, getregion(field, 3).data.parent[:, :, k]; colorrange, colormap)
    if hide_decorations
        hidedecorations!(ax)
    end
    
    # Plot panel 4.
    ax = Axis(fig[2, 3])
    heatmap!(ax, getregion(field, 4).data.parent[:, :, k]; colorrange, colormap)
    if hide_decorations
        hidedecorations!(ax)
    end
    
    # Plot panel 5.
    ax = Axis(fig[1, 3])
    heatmap!(ax, getregion(field, 5).data.parent[:, :, k]; colorrange, colormap)
    if hide_decorations
        hidedecorations!(ax)
    end
    
    # Plot panel 6.
    ax = Axis(fig[1, 4])
    heatmap!(ax, getregion(field, 6).data.parent[:, :, k]; colorrange, colormap)
    if hide_decorations
        hidedecorations!(ax)
    end
    
    # Save figure.
    figure_name = profile * "_panel_wise_visualization_" * field_name * "_k_" * string(k) * ".png"
    save(figure_name, fig) 
    
end

function multi_region_cubed_sphere_plots()

    Nx, Ny, Nz = 5, 5, 1
    cubed_sphere_radius = 1
    grid = ConformalCubedSphereGrid(panel_size=(Nx, Ny, Nz), z=(-1, 0), radius=cubed_sphere_radius, 
                                    horizontal_direction_halo = 1, z_topology=Bounded)
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz
    
    c = CenterField(grid)
     
    initial_tracer_profile = "latitude"
    # Choose initial_tracer_profile to be "latitude" or "longitude".
    
    if initial_tracer_profile == "latitude"
        
        set!(c, (λ, φ, z) -> φ)
        colorrange = (-90, 90)
        
    elseif initial_tracer_profile == "longitude"
    
        set!(c, (λ, φ, z) -> λ)
        colorrange = (-180, 180)
        
    end
    
    colormap = :balance
    fill_halo_regions!(c)
    
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
    heatsphere!(ax, c; colorrange, colormap)
    save(initial_tracer_profile * "_heatsphere_c" * ".png", fig)
    
    fig = Figure()
    ax = Axis(fig[1, 1])
    heatlatlon!(ax, c; colorrange, colormap)
    save(initial_tracer_profile * "_heatlatlon_c" * ".png", fig)

    fig = Figure(resolution = (1200, 600))
    ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
    heatlatlon!(ax, c; colorrange, colormap)
    save(initial_tracer_profile * "_geolatlon_c" * ".png", fig)
    
    k = grid.region_grids[1].Hz + (Nz + 1)÷2
    panel_wise_visualization(initial_tracer_profile, c, "c", k)
    
    u = XFaceField(grid)
    v = YFaceField(grid)
    
    initial_velocity_profile = "latitude"
    # Choose initial_velocity_profile to be "latitude" or "longitude" or "constant" or "solid_body_rotation".
    
    if initial_velocity_profile == "latitude"
    
        for region in 1:6
        
            for i in 1-Hx:Nx+Hx, j in 1-Hy:Ny+Hy
                getregion(u, region).data[i, j, 1] = φnodes(getregion(grid, region), Face(), Center(), Center(); 
                                                            with_halos=true)[i, j, 1]
                getregion(v, region).data[i, j, 1] = φnodes(getregion(grid, region), Center(), Face(), Center(); 
                                                            with_halos=true)[i, j, 1]
            end
    
        end

        colorrange = (-90, 90)
        
    elseif initial_velocity_profile == "longitude"
    
        for region in 1:6
        
            for i in 1-Hx:Nx+Hx, j in 1-Hy:Ny+Hy
                getregion(u, region).data[i, j, 1] = λnodes(getregion(grid, region), Face(), Center(), Center(); 
                                                            with_halos=true)[i, j, 1]
                getregion(v, region).data[i, j, 1] = λnodes(getregion(grid, region), Center(), Face(), Center(); 
                                                            with_halos=true)[i, j, 1]
            end
    
        end

        colorrange = (-180, 180)
        
    elseif initial_velocity_profile == "constant"
    
        u₀ = 10
        v₀ = 100
        U(λ, φ, z) = u₀ 
        V(λ, φ, z) = v₀
        u₀_v₀_abs_max = max(abs(u₀), abs(v₀))
        colorrange = (-u₀_v₀_abs_max, u₀_v₀_abs_max)
    
    elseif initial_velocity_profile == "solid_body_rotation"
    
        time_period = 10
        u_advection = (2π * cubed_sphere_radius)/time_period
        U(λ, φ, z) =   u_advection * cosd(λ) * sind(φ)
        V(λ, φ, z) = - u_advection * sind(λ)
        colorrange = (-u_advection, u_advection)
        
    end
    
    if !(initial_velocity_profile == "latitude" || initial_velocity_profile == "longitude")
    
        set!(u, U)
        set!(v, V)
        
        for _ in 1:2
            fill_halo_regions!(u)
            fill_halo_regions!(v)
            @apply_regionally replace_horizontal_velocity_halos!((; u = u, v = v, w = nothing), grid)
        end
        
    end
    
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
    heatsphere!(ax, u; colorrange, colormap)
    save(initial_velocity_profile * "_heatsphere_u" * ".png", fig)
    
    fig = Figure()
    ax = Axis(fig[1, 1])
    heatlatlon!(ax, u; colorrange, colormap)
    save(initial_velocity_profile * "_heatlatlon_u" * ".png", fig)

    fig = Figure(resolution = (1200, 600))
    ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
    heatlatlon!(ax, u; colorrange, colormap)
    save(initial_velocity_profile * "_geolatlon_u" * ".png", fig)
    
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
    heatsphere!(ax, v; colorrange, colormap)
    save(initial_velocity_profile * "_heatsphere_v" * ".png", fig)
    
    fig = Figure()
    ax = Axis(fig[1, 1])
    heatlatlon!(ax, v; colorrange, colormap)
    save(initial_velocity_profile * "_heatlatlon_v" * ".png", fig)

    fig = Figure(resolution = (1200, 600))
    ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
    heatlatlon!(ax, v; colorrange, colormap)
    save(initial_velocity_profile * "_geolatlon_v" * ".png", fig)  
    
    k = grid.region_grids[1].Hz + (Nz + 1)÷2
    
    panel_wise_visualization(initial_velocity_profile, u, "u", k)
    panel_wise_visualization(initial_velocity_profile, v, "v", k) 
    
end

test_multi_region_cubed_sphere_plots = false
if test_multi_region_cubed_sphere_plots
    multi_region_cubed_sphere_plots()
end