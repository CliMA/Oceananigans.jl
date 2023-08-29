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

function panel_wise_visualization(field_type, field, field_name, function_type, k; # Here k represents the vertical index of the field.
                                  hide_decorations = true, extrema_reduction_factor = 1.0)
    
    if field_type == "zonal_velocity"
        title_part = "u at fca"
    elseif field_type == "meridional_velocity"
        title_part = "v at cfa"
    elseif field_type == "divergence"
        title_part = "u_x + v_y at cca"
    elseif field_name == "c"
        title_part = "cca"
    elseif field_name == "u"
        title_part = "fca"
    elseif field_name == "v"
        title_part = "cfa"
    end
    
    # Create figure.
    fig = Figure(resolution = (1800, 1200))
    colorrange = (minimum(field) * extrema_reduction_factor, maximum(field) * extrema_reduction_factor)
    colormap = :balance
    
    # Plot Panel 1.
    ax_1 = Axis(fig[3,1]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 1: " * title_part, titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_1 = heatmap!(ax_1, getregion(field, 1).data.parent[:, :, k]; colorrange, colormap)
    Colorbar(fig[3,2], hm_1)
    
    # Plot Panel 2.
    ax_2 = Axis(fig[3,3]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 2: " * title_part, titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_2 = heatmap!(ax_2, getregion(field, 2).data.parent[:, :, k]; colorrange, colormap)
    Colorbar(fig[3,4], hm_2)
    
    # Plot Panel 3.
    ax_3 = Axis(fig[2,3]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 3: " * title_part, titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_3 = heatmap!(ax_3, getregion(field, 3).data.parent[:, :, k]; colorrange, colormap)
    Colorbar(fig[2,4], hm_3)    
    
    # Plot Panel 4.
    ax_4 = Axis(fig[2,5]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 4: " * title_part, titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_4 = heatmap!(ax_4, getregion(field, 4).data.parent[:, :, k]; colorrange, colormap)
    Colorbar(fig[2,6], hm_4)       

    # Plot Panel 5.
    ax_5 = Axis(fig[1,5]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 5: " * title_part, titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_5 = heatmap!(ax_5, getregion(field, 5).data.parent[:, :, k]; colorrange, colormap)
    Colorbar(fig[1,6], hm_5)        

    # Plot Panel 6.
    ax_6 = Axis(fig[1,7]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 6: " * title_part, titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_6 = heatmap!(ax_6, getregion(field, 6).data.parent[:, :, k]; colorrange, colormap)
    Colorbar(fig[1,8], hm_6)    
    
    if hide_decorations
        hidedecorations!(ax_1)
        hidedecorations!(ax_2)
        hidedecorations!(ax_3)
        hidedecorations!(ax_4)
        hidedecorations!(ax_5)
        hidedecorations!(ax_6)
    end
    
    # Save figure.
    figure_name = field_type * "_panel_wise_visualization_" * field_name * "_" * function_type * ".png"
    save(figure_name, fig) 
    
end

function multi_region_cubed_sphere_latitude_longitude_plots(coordinate_type, function_type, set_function_fill_halos)

    Nx, Ny, Nz = 5, 5, 1
    cubed_sphere_radius = 1
    grid = ConformalCubedSphereGrid(panel_size=(Nx, Ny, Nz), z=(-1, 0), radius=cubed_sphere_radius, 
                                    horizontal_direction_halo = 1, z_topology=Bounded)
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz
    
    c = CenterField(grid)
    u = XFaceField(grid)
    v = YFaceField(grid)

    if coordinate_type == "latitude"
        
        if function_type == "set"
        
            set!(c, (λ, φ, z) -> φ)
            set!(u, (λ, φ, z) -> φ)
            set!(v, (λ, φ, z) -> φ)
            
            if set_function_fill_halos
                fill_halo_regions!(c)
                fill_halo_regions!(u)
                fill_halo_regions!(v)
            end
            
        elseif function_type == "nodes"
        
            for region in 1:6
                for i in 1-Hx:Nx+Hx, j in 1-Hy:Ny+Hy
                    getregion(c, region).data[i, j, 1] = φnodes(getregion(grid, region), Center(), Center(), Center();
                                                                with_halos=true)[i, j, 1]
                    getregion(u, region).data[i, j, 1] = φnodes(getregion(grid, region), Face(), Center(), Center(); 
                                                                with_halos=true)[i, j, 1]
                    getregion(v, region).data[i, j, 1] = φnodes(getregion(grid, region), Center(), Face(), Center(); 
                                                                with_halos=true)[i, j, 1]
                end
            end
            
        end

        colorrange = (-90, 90)
        
    elseif coordinate_type == "longitude"
    
        if function_type == "set"
        
            set!(c, (λ, φ, z) -> λ)
            set!(u, (λ, φ, z) -> λ)
            set!(v, (λ, φ, z) -> λ)
            
            if set_function_fill_halos
                fill_halo_regions!(c)
                fill_halo_regions!(u)
                fill_halo_regions!(v)
            end
            
        elseif function_type == "nodes"
        
            for region in 1:6
                for i in 1-Hx:Nx+Hx, j in 1-Hy:Ny+Hy
                    getregion(c, region).data[i, j, 1] = λnodes(getregion(grid, region), Center(), Center(), Center();
                                                                with_halos=true)[i, j, 1]
                    getregion(u, region).data[i, j, 1] = λnodes(getregion(grid, region), Face(), Center(), Center(); 
                                                                with_halos=true)[i, j, 1]
                    getregion(v, region).data[i, j, 1] = λnodes(getregion(grid, region), Center(), Face(), Center(); 
                                                                with_halos=true)[i, j, 1]
                end
            end
            
        end

        colorrange = (-180, 180)
        
    end
    
    k = grid.region_grids[1].Hz + (Nz + 1)÷2
    
    panel_wise_visualization(coordinate_type, c, "c", function_type, k)
    panel_wise_visualization(coordinate_type, u, "u", function_type, k)
    panel_wise_visualization(coordinate_type, v, "v", function_type, k)
    
    colormap = :balance
    
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
    heatsphere!(ax, c; colorrange, colormap)
    save(coordinate_type * "_heatsphere_c" * "_" * function_type * ".png", fig)
    
    fig = Figure()
    ax = Axis(fig[1, 1])
    heatlatlon!(ax, c; colorrange, colormap)
    save(coordinate_type * "_heatlatlon_c" * "_" * function_type * ".png", fig)

    fig = Figure(resolution = (1200, 600))
    ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
    heatlatlon!(ax, c; colorrange, colormap)
    save(coordinate_type * "_geolatlon_c" * "_" * function_type * ".png", fig)
    
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
    heatsphere!(ax, u; colorrange, colormap)
    save(coordinate_type * "_heatsphere_u" * "_" * function_type * ".png", fig)
    
    fig = Figure()
    ax = Axis(fig[1, 1])
    heatlatlon!(ax, u; colorrange, colormap)
    save(coordinate_type * "_heatlatlon_u" * "_" * function_type * ".png", fig)

    fig = Figure(resolution = (1200, 600))
    ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
    heatlatlon!(ax, u; colorrange, colormap)
    save(coordinate_type * "_geolatlon_u" * "_" * function_type * ".png", fig)
    
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=(1, 1, 1), limits=((-1, 1), (-1, 1), (-1, 1)))
    heatsphere!(ax, v; colorrange, colormap)
    save(coordinate_type * "_heatsphere_v" * "_" * function_type * ".png", fig)
    
    fig = Figure()
    ax = Axis(fig[1, 1])
    heatlatlon!(ax, v; colorrange, colormap)
    save(coordinate_type * "_heatlatlon_v" * "_" * function_type * ".png", fig)

    fig = Figure(resolution = (1200, 600))
    ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
    heatlatlon!(ax, v; colorrange, colormap)
    save(coordinate_type * "_geolatlon_v" * "_" * function_type * ".png", fig)  
    
end

test_multi_region_cubed_sphere_plots = false
if test_multi_region_cubed_sphere_plots
    coordinate_type = "longitude" # Choose between "latitude" and "longitude".
    function_type = "nodes" # Choose between "set" and "nodes".
    set_function_fill_halos = false # Choose between true and false.
    multi_region_cubed_sphere_latitude_longitude_plots(coordinate_type, function_type, set_function_fill_halos)
end

function tangential_velocities_around_cubed_sphere_panels(u, v, k; # Here k represents the vertical index of the field.
                                                          hide_decorations = true, extrema_reduction_factor = 1.0)
    
    ## Plot tangential velocity around cubed sphere panels 1, 2, 4, and 5.
    
    # Create figure.
    fig = Figure(resolution = (3000, 600))
    colorrange_min = min(minimum(u) * extrema_reduction_factor, minimum(v) * extrema_reduction_factor)
    colorrange_max = max(maximum(u) * extrema_reduction_factor, maximum(v) * extrema_reduction_factor)
    colorrange = (colorrange_min, colorrange_max)
    colormap = :balance

    # Plot Panel 1.
    ax_1 = Axis(fig[1,1]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 1", titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_1 = heatmap!(ax_1, getregion(u, 1).data.parent[:, :, k]; colorrange, colormap)
    Colorbar(fig[1,2], hm_1)

    # Plot Panel 2.
    ax_2 = Axis(fig[1,3]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 2", titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_2 = heatmap!(ax_2, getregion(u, 2).data.parent[:, :, k]; colorrange, colormap)
    Colorbar(fig[1,4], hm_2)   

    # Plot Panel 4.
    ax_4 = Axis(fig[1,5]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 4", titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_4 = heatmap!(ax_4, rotr90(getregion(v, 4).data.parent[:, :, k]); colorrange, colormap)
    Colorbar(fig[1,6], hm_4)       

    # Plot Panel 5.
    ax_5 = Axis(fig[1,7]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 5", titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_5 = heatmap!(ax_5, rotr90(getregion(v, 5).data.parent[:, :, k]); colorrange, colormap)
    Colorbar(fig[1,8], hm_5)        

    if hide_decorations
        hidedecorations!(ax_1)
        hidedecorations!(ax_2)
        hidedecorations!(ax_4)
        hidedecorations!(ax_5)
    end

    # Save figure.
    figure_name = "tangential_velocity_around_cubed_sphere_panels_1245.png"
    save(figure_name, fig) 
    
    ## Plot tangential velocity around cubed sphere panels 2, 3, 5, and 6.
    
    # Create figure.
    fig = Figure(resolution = (3000, 600))
    colorrange_min = min(minimum(u) * extrema_reduction_factor, minimum(v) * extrema_reduction_factor)
    colorrange_max = max(maximum(u) * extrema_reduction_factor, maximum(v) * extrema_reduction_factor)
    colorrange = (colorrange_min, colorrange_max)
    colormap = :balance

    # Plot Panel 2.
    ax_2 = Axis(fig[1,1]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 2", titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_2 = heatmap!(ax_2, rotr90(getregion(v, 2).data.parent[:, :, k]); colorrange, colormap)
    Colorbar(fig[1,2], hm_2)   
    
    # Plot Panel 3.
    ax_3 = Axis(fig[1,3]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 3", titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_3 = heatmap!(ax_3, rotr90(getregion(v, 3).data.parent[:, :, k]); colorrange, colormap)
    Colorbar(fig[1,4], hm_3)  
    
    # Plot Panel 5.
    ax_5 = Axis(fig[1,5]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
    ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
    aspect = 1.0, title = "Panel 5", titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_5 = heatmap!(ax_5, getregion(u, 5).data.parent[:, :, k]; colorrange, colormap)
    Colorbar(fig[1,6], hm_5)        

    # Plot Panel 6.
    ax_6 = Axis(fig[1,7]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 6", titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_6 = heatmap!(ax_6, getregion(u, 6).data.parent[:, :, k]; colorrange, colormap)
    Colorbar(fig[1,8], hm_6)  
    
    if hide_decorations
        hidedecorations!(ax_2)
        hidedecorations!(ax_3)
        hidedecorations!(ax_5)
        hidedecorations!(ax_6)
    end

    # Save figure.
    figure_name = "tangential_velocity_around_cubed_sphere_panels_2356.png"
    save(figure_name, fig) 
    
    ## Plot tangential velocity around cubed sphere panels 6, 1, 3, and 4.
    
    # Create figure.
    fig = Figure(resolution = (3000, 600))
    colorrange_min = min(minimum(u) * extrema_reduction_factor, minimum(v) * extrema_reduction_factor)
    colorrange_max = max(maximum(u) * extrema_reduction_factor, maximum(v) * extrema_reduction_factor)
    colorrange = (colorrange_min, colorrange_max)
    colormap = :balance

    # Plot Panel 6.
    ax_6 = Axis(fig[1,1]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 6", titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_6 = heatmap!(ax_6, rotr90(getregion(v, 6).data.parent[:, :, k]); colorrange, colormap)
    Colorbar(fig[1,2], hm_6)   
    
    # Plot Panel 1.
    ax_1 = Axis(fig[1,3]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 1", titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_1 = heatmap!(ax_1, rotr90(getregion(v, 1).data.parent[:, :, k]); colorrange, colormap)
    Colorbar(fig[1,4], hm_1)  
    
    # Plot Panel 3.
    ax_3 = Axis(fig[1,5]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 3", titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_3 = heatmap!(ax_3, getregion(u, 3).data.parent[:, :, k]; colorrange, colormap)
    Colorbar(fig[1,6], hm_3)        

    # Plot Panel 4.
    ax_4 = Axis(fig[1,7]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, 
                ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
                aspect = 1.0, title = "Panel 4", titlesize = 27.5, titlegap = 15, titlefont = :bold)
    hm_4 = heatmap!(ax_4, getregion(u, 4).data.parent[:, :, k]; colorrange, colormap)
    Colorbar(fig[1,8], hm_4)  
    
    if hide_decorations
        hidedecorations!(ax_6)
        hidedecorations!(ax_1)
        hidedecorations!(ax_3)
        hidedecorations!(ax_4)
    end

    # Save figure.
    figure_name = "tangential_velocity_around_cubed_sphere_panels_6134.png"
    save(figure_name, fig) 

end  

function normal_velocities_around_cubed_sphere_panels(u, v, k; # Here k represents the vertical index of the field.
                                                      hide_decorations = true, extrema_reduction_factor = 1.0)

## Plot normal velocity around cubed sphere panels 1, 2, 4, and 5.

# Create figure.
fig = Figure(resolution = (3000, 600))
colorrange_min = min(minimum(u) * extrema_reduction_factor, minimum(v) * extrema_reduction_factor)
colorrange_max = max(maximum(u) * extrema_reduction_factor, maximum(v) * extrema_reduction_factor)
colorrange = (colorrange_min, colorrange_max)
colormap = :balance

# Plot Panel 1.
ax_1 = Axis(fig[1,1]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, ylabelsize = 22.5, 
            xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1.0, 
            title = "Panel 1", titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_1 = heatmap!(ax_1, getregion(v, 1).data.parent[:, :, k]; colorrange, colormap)
Colorbar(fig[1,2], hm_1)

# Plot Panel 2.
ax_2 = Axis(fig[1,3]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, ylabelsize = 22.5, 
            xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1.0, 
            title = "Panel 2", titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_2 = heatmap!(ax_2, getregion(v, 2).data.parent[:, :, k]; colorrange, colormap)
Colorbar(fig[1,4], hm_2)   

# Plot Panel 4.
ax_4 = Axis(fig[1,5]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, ylabelsize = 22.5, 
            xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1.0, 
            title = "Panel 4", titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_4 = heatmap!(ax_4, -rotr90(getregion(u, 4).data.parent[:, :, k]); colorrange, colormap)
Colorbar(fig[1,6], hm_4)       

# Plot Panel 5.
ax_5 = Axis(fig[1,7]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, ylabelsize = 22.5, 
            xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1.0, 
            title = "Panel 5", titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_5 = heatmap!(ax_5, -rotr90(getregion(u, 5).data.parent[:, :, k]); colorrange, colormap)
Colorbar(fig[1,8], hm_5)        

if hide_decorations
    hidedecorations!(ax_1)
    hidedecorations!(ax_2)
    hidedecorations!(ax_4)
    hidedecorations!(ax_5)
end

# Save figure.
figure_name = "normal_velocity_around_cubed_sphere_panels_1245.png"
save(figure_name, fig) 

## Plot normal velocity around cubed sphere panels 2, 3, 5, and 6.

# Create figure.
fig = Figure(resolution = (3000, 600))
colorrange_min = min(minimum(u) * extrema_reduction_factor, minimum(v) * extrema_reduction_factor)
colorrange_max = max(maximum(u) * extrema_reduction_factor, maximum(v) * extrema_reduction_factor)
colorrange = (colorrange_min, colorrange_max)
colormap = :balance

# Plot Panel 2.
ax_2 = Axis(fig[1,1]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, ylabelsize = 22.5, 
            xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1.0, 
            title = "Panel 2", titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_2 = heatmap!(ax_2, -rotr90(getregion(u, 2).data.parent[:, :, k]); colorrange, colormap)
Colorbar(fig[1,2], hm_2)   

# Plot Panel 3.
ax_3 = Axis(fig[1,3]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, ylabelsize = 22.5, 
            xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1.0, 
            title = "Panel 3", titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_3 = heatmap!(ax_3, -rotr90(getregion(u, 3).data.parent[:, :, k]); colorrange, colormap)
Colorbar(fig[1,4], hm_3)  

# Plot Panel 5.
ax_5 = Axis(fig[1,5]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, ylabelsize = 22.5, 
            xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1.0, 
            title = "Panel 5", titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_5 = heatmap!(ax_5, getregion(v, 5).data.parent[:, :, k]; colorrange, colormap)
Colorbar(fig[1,6], hm_5)        

# Plot Panel 6.
ax_6 = Axis(fig[1,7]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, ylabelsize = 22.5, 
            xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1.0, 
            title = "Panel 6", titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_6 = heatmap!(ax_6, getregion(v, 6).data.parent[:, :, k]; colorrange, colormap)
Colorbar(fig[1,8], hm_6)  

if hide_decorations
    hidedecorations!(ax_2)
    hidedecorations!(ax_3)
    hidedecorations!(ax_5)
    hidedecorations!(ax_6)
end

# Save figure.
figure_name = "normal_velocity_around_cubed_sphere_panels_2356.png"
save(figure_name, fig) 

## Plot normal velocity around cubed sphere panels 6, 1, 3, and 4.

# Create figure.
fig = Figure(resolution = (3000, 600))
colorrange_min = min(minimum(u) * extrema_reduction_factor, minimum(v) * extrema_reduction_factor)
colorrange_max = max(maximum(u) * extrema_reduction_factor, maximum(v) * extrema_reduction_factor)
colorrange = (colorrange_min, colorrange_max)
colormap = :balance

# Plot Panel 6.
ax_6 = Axis(fig[1,1]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, ylabelsize = 22.5, 
            xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1.0, 
            title = "Panel 6", titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_6 = heatmap!(ax_6, -rotr90(getregion(u, 6).data.parent[:, :, k]); colorrange, colormap)
Colorbar(fig[1,2], hm_6)   

# Plot Panel 1.
ax_1 = Axis(fig[1,3]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, ylabelsize = 22.5, 
            xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1.0, 
            title = "Panel 1", titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_1 = heatmap!(ax_1, -rotr90(getregion(u, 1).data.parent[:, :, k]); colorrange, colormap)
Colorbar(fig[1,4], hm_1)  

# Plot Panel 3.
ax_3 = Axis(fig[1,5]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, ylabelsize = 22.5, 
            xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1.0, 
            title = "Panel 3", titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_3 = heatmap!(ax_3, getregion(v, 3).data.parent[:, :, k]; colorrange, colormap)
Colorbar(fig[1,6], hm_3)        

# Plot Panel 4.
ax_4 = Axis(fig[1,7]; xlabel = "Local x direction", ylabel = "Local y direction", xlabelsize = 22.5, ylabelsize = 22.5, 
            xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1.0, 
            title = "Panel 4", titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_4 = heatmap!(ax_4, getregion(v, 4).data.parent[:, :, k]; colorrange, colormap)
Colorbar(fig[1,8], hm_4)  

if hide_decorations
    hidedecorations!(ax_6)
    hidedecorations!(ax_1)
    hidedecorations!(ax_3)
    hidedecorations!(ax_4)
end

# Save figure.
figure_name = "normal_velocity_around_cubed_sphere_panels_6134.png"
save(figure_name, fig) 

end  