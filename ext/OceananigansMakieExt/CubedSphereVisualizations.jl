module CubedSphereVisualizations

using Makie
using Oceananigans
using Oceananigans: location, CubedSphereField
using Oceananigans.MultiRegion: number_of_regions
using Oceananigans.Utils: getregion

using ..Imaginocean

export specify_colorrange, specify_colorrange_time_series, panelwise_visualization, geo_heatmap_visualization,
    panelwise_visualization_animation, geo_heatmap_visualization_animation

function compute_size_metrics(grid::ConformalCubedSphereGridOfSomeKind, field::CubedSphereField, ssh::Bool,
                              consider_all_levels::Bool, levels::UnitRange{Int}, read_parent_field_data::Bool)
    Nx, Ny, Nz = size(field)
    Hx, Hy, Hz = halo_size(grid)

    hx = read_parent_field_data ? Hx : 0
    hy = read_parent_field_data ? Hy : 0
    hz = read_parent_field_data ? Hz : 0
    
    if ssh
        consider_all_levels = false
        levels = read_parent_field_data ? (1:1) : (Nz+1:Nz+1)
    else
        if consider_all_levels
            levels = 1:Nz .+ hz
        else
            levels = levels .+ hz
        end
    end

    return Nx, Ny, Nz, hx, hy, hz, consider_all_levels, levels
end

function compute_vertical_index(grid::ConformalCubedSphereGridOfSomeKind, field::CubedSphereField, k::Int, ssh::Bool,
                                read_parent_field_data::Bool)
    Nz = size(field, 3)
    Hz = halo_size(grid)[3]
    hz = read_parent_field_data ? Hz : 0
    if ssh
        k = read_parent_field_data ? 1 : Nz + 1
    else
        k += hz
    end
    return k
end

function extract_field_time_series_array(grid, field_time_series;
                                         with_halos::Bool = false,
                                         ssh::Bool = false,
                                         consider_all_levels::Bool = true,
                                         levels::UnitRange{Int} = 1:1,
                                         read_parent_field_data::Bool = false,
                                         Δ::Int = 1)
    Nx, Ny, Nz, hx, hy, hz, consider_all_levels, levels = (
        compute_size_metrics(grid, field_time_series[1], ssh, consider_all_levels, levels, read_parent_field_data))
    Hx, Hy, Hz = halo_size(grid)
    
    n = length(field_time_series)
    m = floor(Int, (length(field_time_series) - 1)/Δ + 1)

    if with_halos
        field_time_series_array = zeros(Nx+2Hx, Ny+2Hy, length(levels), 6, m)
    else
        field_time_series_array = zeros(Nx, Ny, length(levels), 6, m)
    end

    j = 0
    for i in 1:Δ:n
        j += 1
        for region in 1:number_of_regions(grid)
            if with_halos
                field_time_series_array[:, :, :, region, j] = (
                    getregion(field_time_series[i], region)[1-Hx+hx:Nx+Hx+hx, 1-Hy+hy:Ny+Hy+hy, levels])
            else
                field_time_series_array[:, :, :, region, j] = (
                    getregion(field_time_series[i], region)[1+hx:Nx+hx, 1+hy:Ny+hy, levels])
            end
        end
    end
    
    return field_time_series_array
end

function interpolate_cubed_sphere_field_to_cell_centers(grid, field, field_location;
                                                        ssh::Bool = false,
                                                        consider_all_levels::Bool = true,
                                                        levels::UnitRange{Int} = 1:1,
                                                        read_parent_field_data::Bool = false)
    if field_location == "cc" && !read_parent_field_data
        return field
    end
    Nx, Ny, Nz, hx, hy, hz, consider_all_levels, levels = compute_size_metrics(
        grid, field, ssh, consider_all_levels, levels, read_parent_field_data)

    interpolated_field = Field{Center, Center, Center}(grid, indices = (:, :, levels))

    set!(interpolated_field, 0)

    @inbounds for region in 1:number_of_regions(grid), j in 1:Ny, i in 1:Nx
        dest = getregion(interpolated_field, region)
        src  = getregion(field, region)

        if field_location == "fc"
            dest[i, j, levels] = 0.5(src[i+hx, j+hy, levels] + src[i+1+hx, j+hy, levels])

        elseif field_location == "cf"
            dest[i, j, levels] = 0.5(src[i+hx, j+hy, levels] + src[i+hx, j+1+hy, levels])

        elseif field_location == "ff"
            dest[i, j, levels] = 0.25(src[i+hx, j+hy, levels] + src[i+1+hx, j+hy, levels] + src[i+1+hx, j+1+hy, levels]
                                      + src[i+hx, j+1+hy, levels])

        elseif field_location == "cc"
            dest[i, j, levels] = src[i+hx, j+hy, levels]
        end
    end

    return interpolated_field
end

function specify_colorrange(grid, field;
                            k::Int = 1,
                            ssh::Bool = false,
                            consider_all_levels::Bool = true,
                            levels::UnitRange{Int} = k:k,
                            read_parent_field_data::Bool = false,
                            use_symmetric_colorrange::Bool = true)
    Nx, Ny, Nz, hx, hy, hz, consider_all_levels, levels = compute_size_metrics(
        grid, field, ssh, consider_all_levels, levels, read_parent_field_data)
    
    field_array = zeros(Nx, Ny, length(levels), 6)
    for region in 1:number_of_regions(grid)
        field_array[:, :, :, region] = getregion(field, region)[1+hx:Nx+hx, 1+hy:Ny+hy, levels]
    end
    
    field_maximum = maximum(field_array)
    field_minimum = minimum(field_array)
    
    if use_symmetric_colorrange
        colorrange_limit = max(abs(field_maximum), abs(field_minimum))
        colorrange = [-colorrange_limit, colorrange_limit]
    else
        colorrange = [field_minimum, field_maximum]
    end
    
    return colorrange
end

function specify_colorrange_time_series(grid, field_time_series;
                                        ssh::Bool = false,
                                        consider_all_levels::Bool = true,
                                        levels::UnitRange{Int} = 1:1,
                                        read_parent_field_data::Bool = false,
                                        Δ::Int = 1,
                                        use_symmetric_colorrange::Bool = true)
    field_time_series_array = (
        extract_field_time_series_array(grid, field_time_series;
                                        ssh, consider_all_levels, levels, read_parent_field_data, Δ))

    field_maximum = maximum(field_time_series_array)
    field_minimum = minimum(field_time_series_array)

    if use_symmetric_colorrange
        colorrange_limit = max(abs(field_maximum), abs(field_minimum))
        colorrange = [-colorrange_limit, colorrange_limit]
    else
        colorrange = [field_minimum, field_maximum]
    end
    
    return colorrange
end

function panelwise_visualization(grid, field;
                                 with_halos::Bool = false,
                                 k::Int = 1,
                                 use_symmetric_colorrange::Bool = true,
                                 ssh::Bool = false,
                                 consider_all_levels::Bool = true,
                                 levels::UnitRange{Int} = k:k,
                                 read_parent_field_data::Bool = false,
                                 colorrange::Union{Nothing, Vector} = nothing,
                                 colormap::Union{Nothing, Symbol} = nothing)
    fig = Figure(size = (2450, 1400))

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xlabelpadding = 10, ylabelpadding = 10, xticklabelsize = 17.5,
                   yticklabelsize = 17.5, xticklabelpad = 20, yticklabelpad = 20, aspect = 1.0, titlesize = 27.5,
                   titlegap = 15, titlefont = :bold, xlabel = "Local x direction", ylabel = "Local y direction")

    if isnothing(colorrange)
        colorrange = specify_colorrange(grid, field;
                                        k, ssh, consider_all_levels, levels, read_parent_field_data,
                                        use_symmetric_colorrange)
    end

    colormap = something(colormap, use_symmetric_colorrange ? :balance : :amp)

    Nx, Ny, Nz, hx, hy, hz, consider_all_levels, levels = compute_size_metrics(
        grid, field, ssh, consider_all_levels, levels, read_parent_field_data)
    k = compute_vertical_index(grid, field, k, ssh, read_parent_field_data)

    function slice_panel_data(f, panel)
        if with_halos
            data = getregion(f, panel)[:, :, k]
        else
            data = getregion(f, panel)[1+hx:Nx+hx, 1+hy:Ny+hy, k]
        end
        return parent(data)
    end

    panel_positions = [(3, 1), (3, 3), (2, 3), (2, 5), (1, 5), (1, 7)]

    for (i, pos) in enumerate(panel_positions)
        ax = Axis(fig[pos...]; title = "Panel $i", axis_kwargs...)
        hm = heatmap!(ax, slice_panel_data(field, i); colorrange, colormap)
        Colorbar(fig[pos[1], pos[2] + 1], hm)
    end

    resize_to_layout!(fig)

    return fig
end

function specify_geo_heatmap_plot_attributes(geo_heatmap_type, title)
    if geo_heatmap_type == "heatlatlon"
        axis_kwargs = (xlabelsize = 37.5, ylabelsize = 37.5, xlabelpadding = 25, ylabelpadding = 25,
                       xticksize = 15, yticksize = 15, xticklabelsize = 32.5, yticklabelsize = 32.5, xticklabelpad = 20,
                       yticklabelpad = 20, titlesize = 45, titlegap = 30, titlefont = :bold)
        fig = Figure(size = (2700, 1300))
        ax = Axis(fig[1, 1]; title, axis_kwargs...)
        colorbar_kwargs = (labelsize = 37.5, labelpadding = 25, ticklabelsize = 30, ticksize = 22.5, width = 35,
                           height = Relative(1))
    elseif geo_heatmap_type == "heatsphere"
        axis_kwargs = (xlabelvisible = false, ylabelvisible = false, zlabelvisible = false, xticksvisible = false,
                       yticksvisible = false, zticksvisible = false, xticklabelsvisible = false,
                       yticklabelsvisible = false, zticklabelsvisible = false, titlesize = 45, titlegap = 30,
                       titlefont = :bold, tellwidth = false, tellheight = false, protrusions = (0, 0, 0, 0),
                       limits = ((-1, 1), (-1, 1), (-1, 1)))
        fig = Figure(size = (1500, 1300), figure_padding = (0, 0, 0, 100)) # (left, right, bottom, top)
        ax = Axis3(fig[1, 1]; title, aspect = :data, perspectiveness = 0, axis_kwargs...)
        colorbar_kwargs = (labelsize = 37.5, labelpadding = 25, ticklabelsize = 30, ticksize = 17.5, width = 25,
                           height = Relative(0.995))
    else
        error("geo_heatmap_type must be either 'heatlatlon' or 'heatsphere'")
    end
    colorbar_column_width_scale = 1
    column_gap = 50
    return fig, ax, colorbar_kwargs, colorbar_column_width_scale, column_gap
end

function geo_heatmap_visualization(grid, field, field_location, title;
                                   geo_heatmap_type::String = "heatlatlon",
                                   k::Int = 1,
                                   use_symmetric_colorrange::Bool = true,
                                   ssh::Bool = false,
                                   consider_all_levels::Bool = true,
                                   levels::UnitRange{Int} = k:k,
                                   read_parent_field_data::Bool = false,
                                   colorbarlabel::String = "",
                                   colorrange::Union{Nothing, Vector} = nothing,
                                   colormap::Union{Nothing, Symbol} = nothing)
    interpolated_field = interpolate_cubed_sphere_field_to_cell_centers(grid, field, field_location;
                                                                        ssh, consider_all_levels, levels,
                                                                        read_parent_field_data)

    if isnothing(colorrange)
        colorrange = specify_colorrange(grid, interpolated_field;
                                        k, ssh, consider_all_levels, levels, read_parent_field_data,
                                        use_symmetric_colorrange)
    end

    colormap = something(colormap, use_symmetric_colorrange ? :balance : :amp)

    fig, ax, colorbar_kwargs, colorbar_column_width_scale, column_gap = (
        specify_geo_heatmap_plot_attributes(geo_heatmap_type, title))
    if geo_heatmap_type == "heatlatlon"
        heatlatlon!(ax, interpolated_field, k; colorrange, colormap)
    elseif geo_heatmap_type == "heatsphere"
        heatsphere!(ax, interpolated_field, k; colorrange, colormap)
    else
        error("geo_heatmap_type must be either 'heatlatlon' or 'heatsphere'")
    end

    Colorbar(fig[1, 2]; limits = colorrange, colormap, label = colorbarlabel, colorbar_kwargs...)
    colsize!(fig.layout, 2, Auto(colorbar_column_width_scale))
    colgap!(fig.layout, column_gap)

    resize_to_layout!(fig)

    return fig
end

function panelwise_visualization_animation_Makie(grid, field_time_series;
                                                 with_halos::Bool = false,
                                                 start_index::Int = 1,
                                                 k::Int = 1,
                                                 use_symmetric_colorrange::Bool = true,
                                                 ssh::Bool = false,
                                                 consider_all_levels::Bool = true,
                                                 levels::UnitRange{Int} = k:k,
                                                 read_parent_field_data::Bool = false,
                                                 Δ::Int = 1,
                                                 colorrange::Union{Nothing, Vector} = nothing,
                                                 colormap::Union{Nothing, Symbol} = nothing,
                                                 framerate::Int = 10,
                                                 output_directory::AbstractString = "output_directory",
                                                 filename::AbstractString = "filename",
                                                 format::AbstractString = ".mp4")
    # Observables
    n = Observable(start_index) # the current index
    field = @lift field_time_series[$n]

    # Create the initial visualization.
    fig = Figure(size = (2450, 1400))

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xlabelpadding = 10, ylabelpadding = 10, xticklabelsize = 17.5,
                   yticklabelsize = 17.5, xticklabelpad = 20, yticklabelpad = 20, aspect = 1.0, titlesize = 27.5,
                   titlegap = 15, titlefont = :bold, xlabel = "Local x direction", ylabel = "Local y direction")

    if isnothing(colorrange)
        colorrange = specify_colorrange_time_series(grid, field_time_series;
                                                    ssh, consider_all_levels, levels, read_parent_field_data, Δ,
                                                    use_symmetric_colorrange)
    end

    colormap = something(colormap, use_symmetric_colorrange ? :balance : :amp)

    Nx, Ny, Nz, hx, hy, hz, consider_all_levels, levels = compute_size_metrics(
        grid, field[], ssh, consider_all_levels, levels, read_parent_field_data)
    k = compute_vertical_index(grid, field[], k, ssh, read_parent_field_data)

    function slice_panel_data(f, panel)
        if with_halos
            data = getregion(f, panel)[:, :, k]
        else
            data = getregion(f, panel)[1+hx:Nx+hx, 1+hy:Ny+hy, k]
        end
        return parent(data)
    end

    panel_positions = [(3, 1), (3, 3), (2, 3), (2, 5), (1, 5), (1, 7)]

    for (i, pos) in enumerate(panel_positions)
        ax = Axis(fig[pos...]; title = "Panel $i", axis_kwargs...)
        hm = heatmap!(ax, slice_panel_data(field[], i); colorrange, colormap)
        Colorbar(fig[pos[1], pos[2] + 1], hm)
    end

    frames = start_index:length(field_time_series)
    CairoMakie.record(fig, joinpath(output_directory, filename * format), frames, framerate = framerate) do i
        print("Plotting frame $i of $(frames[end]) \r")
        field[] = field_time_series[i]
        for (j, pos) in enumerate(panel_positions)
            ax = Axis(fig[pos...]; title = "Panel $j", axis_kwargs...)
            hm = heatmap!(ax, slice_panel_data(field[], j); colorrange, colormap)
            Colorbar(fig[pos[1], pos[2] + 1], hm)
        end
    end
end

function panelwise_visualization_animation_frames(grid, field_time_series;
                                                  with_halos::Bool = false,
                                                  start_index::Int = 1,
                                                  k::Int = 1,
                                                  use_symmetric_colorrange::Bool = true,
                                                  ssh::Bool = false,
                                                  consider_all_levels::Bool = true,
                                                  levels::UnitRange{Int} = k:k,
                                                  read_parent_field_data::Bool = false,
                                                  Δ::Int = 1,
                                                  colorrange::Union{Nothing, Vector} = nothing,
                                                  colormap::Union{Nothing, Symbol} = nothing,
                                                  output_directory::AbstractString = "output_directory",
                                                  filename::AbstractString = "filename",
                                                  format::AbstractString = ".png")
    fig = Figure(size = (2450, 1400))

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xlabelpadding = 10, ylabelpadding = 10, xticklabelsize = 17.5,
                   yticklabelsize = 17.5, xticklabelpad = 20, yticklabelpad = 20, aspect = 1.0, titlesize = 27.5,
                   titlegap = 15, titlefont = :bold, xlabel = "Local x direction", ylabel = "Local y direction")

    if isnothing(colorrange)
        colorrange = specify_colorrange_time_series(grid, field_time_series;
                                                    ssh, consider_all_levels, levels, read_parent_field_data, Δ,
                                                    use_symmetric_colorrange)
    end

    colormap = something(colormap, use_symmetric_colorrange ? :balance : :amp)

    field = field_time_series[1]
    Nx, Ny, Nz, hx, hy, hz, consider_all_levels, levels = compute_size_metrics(
        grid, field, ssh, consider_all_levels, levels, read_parent_field_data)
    k = compute_vertical_index(grid, field, k, ssh, read_parent_field_data)

    function slice_panel_data(f, panel)
        if with_halos
            data = getregion(f, panel)[:, :, k]
        else
            data = getregion(f, panel)[1+hx:Nx+hx, 1+hy:Ny+hy, k]
        end
        return parent(data)
    end

    panel_positions = [(3, 1), (3, 3), (2, 3), (2, 5), (1, 5), (1, 7)]

    frames = start_index:length(field_time_series)
    for i in frames
        print("Plotting frame $i of $(frames[end]) \r")
        field = field_time_series[i]
        for (j, pos) in enumerate(panel_positions)
            ax = Axis(fig[pos...]; title = "Panel $j", axis_kwargs...)
            hm = heatmap!(ax, slice_panel_data(field, j); colorrange, colormap)
            Colorbar(fig[pos[1], pos[2] + 1], hm)
        end
        save(joinpath(output_directory, filename * "_$i" * format), fig)
    end
end

function panelwise_visualization_animation(grid, field_time_series;
                                           plot_frames::Bool = false,
                                           with_halos::Bool = false,
                                           start_index::Int = 1,
                                           k::Int = 1,
                                           use_symmetric_colorrange::Bool = true,
                                           ssh::Bool = false,
                                           consider_all_levels::Bool = true,
                                           levels::UnitRange{Int} = k:k,
                                           read_parent_field_data::Bool = false,
                                           Δ::Int = 1,
                                           colorrange::Union{Nothing, Vector} = nothing,
                                           colormap::Union{Nothing, Symbol} = nothing,
                                           framerate::Int = 10,
                                           output_directory::AbstractString = "output_directory",
                                           filename::AbstractString = "filename",
                                           figure_format::AbstractString = ".png",
                                           movie_format::AbstractString = ".mp4")
    if plot_frames
        format = figure_format
        path = joinpath(pwd(), output_directory * "/" * filename * "_frames")
        isdir(path) || mkdir(path)
        filename = joinpath(filename * "_frames/", filename)
        panelwise_visualization_animation_frames(grid, field_time_series;
                                                 with_halos, start_index, k, use_symmetric_colorrange, ssh,
                                                 consider_all_levels, levels, read_parent_field_data, Δ, colorrange,
                                                 colormap, output_directory, filename, format)
    else
        format = movie_format
        panelwise_visualization_animation_Makie(grid, field_time_series;
                                                with_halos, start_index, k, use_symmetric_colorrange, ssh,
                                                consider_all_levels, levels, read_parent_field_data, Δ, colorrange,
                                                colormap, framerate, output_directory, filename, format)
    end
end

function geo_heatmap_visualization_animation_Makie(grid, field_time_series, field_location, prettytimes, title_prefix;
                                                   geo_heatmap_type::String = "heatlatlon",
                                                   start_index::Int = 1,
                                                   k::Int = 1,
                                                   use_symmetric_colorrange::Bool = true,
                                                   ssh::Bool = false,
                                                   consider_all_levels::Bool = true,
                                                   levels::UnitRange{Int} = k:k,
                                                   read_parent_field_data::Bool = false,
                                                   Δ::Int = 1,
                                                   colorrange::Union{Nothing, Vector} = nothing,
                                                   colormap::Union{Nothing, Symbol} = nothing,
                                                   colorbarlabel::String = "",
                                                   framerate::Int = 10,
                                                   output_directory::AbstractString = "output_directory",
                                                   filename::AbstractString = "filename",
                                                   format::AbstractString = ".mp4")
    # Observables
    n = Observable(start_index) # the current index
    field = @lift field_time_series[$n]
    prettytime = @lift prettytimes[$n]

    if isnothing(colorrange)
        colorrange = specify_colorrange_time_series(grid, field_time_series;
                                                    ssh, consider_all_levels, levels, read_parent_field_data, Δ,
                                                    use_symmetric_colorrange)
    end

    colormap = something(colormap, use_symmetric_colorrange ? :balance : :amp)

    interpolated_field = interpolate_cubed_sphere_field_to_cell_centers(grid, field[], field_location;
                                                                        ssh, consider_all_levels, levels,
                                                                        read_parent_field_data)

    # Create the initial visualization.
    fig, ax = specify_geo_heatmap_plot_attributes(geo_heatmap_type, title_prefix)
    ax.title = title_prefix * " after " * prettytime[]
    if geo_heatmap_type == "heatlatlon"
        heatlatlon!(ax, interpolated_field, k; colorrange, colormap)
    elseif geo_heatmap_type == "heatsphere"
        heatsphere!(ax, interpolated_field, k; colorrange, colormap)
    else
        error("geo_heatmap_type must be either 'heatlatlon' or 'heatsphere'")
    end

    Colorbar(fig[1, 2]; limits = colorrange, colormap, label = colorbarlabel, labelsize = 37.5, labelpadding = 25,
             ticklabelsize = 30, ticksize = 22.5, width = 35, height = Relative(1))
    colsize!(fig.layout, 1, Auto(0.8))
    colgap!(fig.layout, 75)

    frames = start_index:length(field_time_series)
    CairoMakie.record(fig, joinpath(output_directory, filename * format), frames, framerate = framerate) do i
        print("Plotting frame $i of $(frames[end]) \r")

        field[] = field_time_series[i]
        prettytime[] = prettytimes[i]

        # Update the title of the plot.
        ax.title = title_prefix * " after " * prettytime[]

        # Update the plot.
        interpolated_field = interpolate_cubed_sphere_field_to_cell_centers(grid, field[], field_location;
                                                                            ssh, consider_all_levels, levels,
                                                                            read_parent_field_data)
        if geo_heatmap_type == "heatlatlon"
            heatlatlon!(ax, interpolated_field, k; colorrange, colormap)
        elseif geo_heatmap_type == "heatsphere"
            heatsphere!(ax, interpolated_field, k; colorrange, colormap)
        else
            error("geo_heatmap_type must be either 'heatlatlon' or 'heatsphere'")
        end

        Colorbar(fig[1, 2]; limits = colorrange, colormap, label = colorbarlabel, labelsize = 37.5, labelpadding = 25,
                 ticklabelsize = 30, ticksize = 22.5, width = 35, height = Relative(1))
        colsize!(fig.layout, 1, Auto(0.8))
        colgap!(fig.layout, 75)
    end
end

function geo_heatmap_visualization_animation_frames(grid, field_time_series, field_location, prettytimes, title_prefix;
                                                    geo_heatmap_type::String = "heatlatlon",
                                                    start_index::Int = 1,
                                                    k::Int = 1,
                                                    use_symmetric_colorrange::Bool = true,
                                                    ssh::Bool = false,
                                                    consider_all_levels::Bool = true,
                                                    levels::UnitRange{Int} = k:k,
                                                    read_parent_field_data::Bool = false,
                                                    Δ::Int = 1,
                                                    colorrange::Union{Nothing, Vector} = nothing,
                                                    colormap::Union{Nothing, Symbol} = nothing,
                                                    colorbarlabel::String = "",
                                                    output_directory::AbstractString = "output_directory",
                                                    filename::AbstractString = "filename",
                                                    format::AbstractString = ".png")
    if isnothing(colorrange)
        colorrange = specify_colorrange_time_series(grid, field_time_series;
                                                    ssh, consider_all_levels, levels, read_parent_field_data, Δ,
                                                    use_symmetric_colorrange)
    end

    colormap = something(colormap, use_symmetric_colorrange ? :balance : :amp)

    fig, ax = specify_geo_heatmap_plot_attributes(geo_heatmap_type, title_prefix)

    Colorbar(fig[1, 2]; limits = colorrange, colormap, label = colorbarlabel, labelsize = 37.5, labelpadding = 25,
             ticklabelsize = 30, ticksize = 22.5, width = 35, height = Relative(1))
    colsize!(fig.layout, 1, Auto(0.8))
    colgap!(fig.layout, 75)

    frames = start_index:length(field_time_series)
    for i in frames
        print("Plotting frame $i of $(frames[end]) \r")
        field = field_time_series[i]
        prettytime = prettytimes[i]
        ax.title = title_prefix * " after " * prettytime
        interpolated_field = interpolate_cubed_sphere_field_to_cell_centers(grid, field, field_location;
                                                                            ssh, consider_all_levels, levels,
                                                                            read_parent_field_data)
        if geo_heatmap_type == "heatlatlon"
            heatlatlon!(ax, interpolated_field, k; colorrange, colormap)
        elseif geo_heatmap_type == "heatsphere"
            heatsphere!(ax, interpolated_field, k; colorrange, colormap)
        else
            error("geo_heatmap_type must be either 'heatlatlon' or 'heatsphere'")
        end
        save(joinpath(output_directory, filename * "_$i" * format), fig)
    end
end

function geo_heatmap_visualization_animation(grid, field_time_series, field_location, prettytimes, title_prefix;
                                             geo_heatmap_type::String = "heatlatlon",
                                             plot_frames::Bool = false,
                                             start_index::Int = 1,
                                             k::Int = 1,
                                             use_symmetric_colorrange::Bool = true,
                                             ssh::Bool = false,
                                             consider_all_levels::Bool = true,
                                             levels::UnitRange{Int} = k:k,
                                             read_parent_field_data::Bool = false,
                                             Δ::Int = 1,
                                             colorrange::Union{Nothing, Vector} = nothing,
                                             colormap::Union{Nothing, Symbol} = nothing,
                                             colorbarlabel::String = "",
                                             framerate::Int = 10,
                                             output_directory::AbstractString = "output",
                                             filename::AbstractString = "filename",
                                             figure_format::AbstractString = ".png",
                                             movie_format::AbstractString = ".mp4")
    if plot_frames
        format = figure_format
        path = joinpath(pwd(), output_directory * "/" * filename * "_frames")
        isdir(path) || mkdir(path)
        filename = joinpath(filename * "_frames/", filename)
        geo_heatmap_visualization_animation_frames(grid, field_time_series, field_location, prettytimes, title_prefix;
                                                   geo_heatmap_type, start_index, k, use_symmetric_colorrange, ssh,
                                                   consider_all_levels, levels, read_parent_field_data, Δ,
                                                   colorrange, colormap, colorbarlabel, output_directory, filename,
                                                   format)
    else
        format = movie_format
        geo_heatmap_visualization_animation_Makie(grid, field_time_series, field_location, prettytimes, title_prefix;
                                                  geo_heatmap_type, start_index, k, use_symmetric_colorrange, ssh,
                                                  consider_all_levels, levels, read_parent_field_data, Δ, colorrange,
                                                  colormap, colorbarlabel, framerate, output_directory, filename,
                                                  format)
    end
end

end # module CubedSphereVisualizations
