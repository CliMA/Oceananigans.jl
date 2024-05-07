using LinearAlgebra
using Oceananigans.MultiRegion: getregion
using CairoMakie, GeoMakie, Imaginocean

function interpolate_cubed_sphere_field_to_cell_centers(grid, field, location; k = 1)
    if location == "cc"
        return field
    end
    
    Nx = grid.Nx
    Ny = grid.Ny

    interpolated_field = Field{Center, Center, Center}(grid)

    for region in 1:number_of_regions(grid)
        for j in 1:Ny
            for i in 1:Nx
                if location == "fc"
                    interpolated_field[region][i, j, k] = 0.5(field[region][i, j, k] + field[region][i + 1, j, k])
                elseif location == "cf"
                    interpolated_field[region][i, j, k] = 0.5(field[region][i, j, k] + field[region][i, j + 1, k])
                elseif location == "ff"
                    interpolated_field[region][i, j, k] = 0.25(field[region][i, j, k] + field[region][i + 1, j, k]
                                                               + field[region][i + 1, j + 1, k] + field[region][i, j + 1, k])
                end
            end
        end
    end

    return interpolated_field
end

function make_single_line_or_scatter_plot(output_directory, plot_type, x, y, labels, title, file_name, resolution, 
                                          linewidth, linecolor, marker, markersize, labelsizes, ticklabelsizes, 
                                          labelpaddings, aspect, titlesize, titlegap)
    cwd = pwd()
    path = joinpath(cwd, output_directory)
    if !isdir(path) 
        mkdir(path) 
    end
    cd(path)

    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1]; xlabel = labels[1], ylabel = labels[2], xlabelsize = labelsizes[1], ylabelsize = labelsizes[2], 
    xticklabelsize = ticklabelsizes[1], yticklabelsize = ticklabelsizes[2], xlabelpadding = labelpaddings[1], 
    ylabelpadding = labelpaddings[2], aspect = aspect, title = title, titlesize = titlesize, 
    titlegap = titlegap, titlefont = :bold)

    if plot_type == "line_plot"
        lines!(ax, x, y, linewidth = linewidth, color = linecolor)
    elseif plot_type == "scatter_plot"
        scatter!(ax, x, y, marker = marker, markersize = markersize, color = linecolor)
    elseif plot_type == "scatter_line_plot"
        scatterlines!(ax, x, y, linewidth = linewidth, marker = marker, markersize = markersize, color = linecolor)
    end

    save(file_name, fig)
    cd(cwd)
end

function make_heat_map_or_contour_plot(output_directory, plot_type, x, y, φ, φ_limits, file_name, labels, title,
                                       resolution, labelsizes, ticklabelsizes, labelpaddings, aspect, titlesize, 
                                       titlegap, colormap, contourlevels; specify_axis_limits = true, 
                                       use_specified_limits = false)
    cwd = pwd()
    path = joinpath(cwd, output_directory)
    if !isdir(path) 
        mkdir(path) 
    end
    cd(path)

    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1]; xlabel = labels[1], ylabel = labels[2], xlabelsize = labelsizes[1], ylabelsize = labelsizes[2], 
    xticklabelsize = ticklabelsizes[1], yticklabelsize = ticklabelsizes[2], xlabelpadding = labelpaddings[1], 
    ylabelpadding = labelpaddings[2], aspect = aspect, title = title, titlesize = titlesize, 
    titlegap = titlegap, titlefont = :bold)

    if specify_axis_limits
        xlims!(ax, (minimum(x), maximum(x)))
        ylims!(ax, (minimum(y), maximum(y)))
    end

    if !use_specified_limits
        φ_limits = [minimum(φ), maximum(φ)]
    end

    if plot_type == "heat_map"
        hm = heatmap!(ax, x, y, φ; colorrange = φ_limits, colormap = colormap)
    elseif plot_type == "filled_contour_plot"
        hm = contourf!(ax, x, y, φ; levels = range(φ_limits..., length=contourlevels), colormap = colormap)  
    end

    Colorbar(fig[1,2], hm)

    save(file_name, fig)
    cd(cwd)
end

function specify_colorrange(φ; use_symmetric_colorrange = true)
    φ_maximum = maximum(φ)
    φ_minimum = minimum(φ)
    if use_symmetric_colorrange
        colorrange_limit = max(abs(φ_maximum), abs(φ_minimum))
        colorrange = (-colorrange_limit, colorrange_limit)
    else
        colorrange = (φ_minimum, φ_maximum)
    end
    return colorrange
end

function specify_colorrange(grid, φ; use_symmetric_colorrange = true, ssh = false)
    Nx = grid.Nx
    Ny = grid.Ny
    Nz = grid.Nz
    
    if ssh
        φ_array = zeros(Nx, Ny, 1, 6)
        φ_array_vertical_dimension_limits = Nz+1:Nz+1
    else
        φ_array = zeros(Nx, Ny, Nz, 6)
        φ_array_vertical_dimension_limits = 1:Nz
    end
    
    for region in 1:6
        φ_array[:, :, :, region] = φ[region].data[1:Nx, 1:Ny, φ_array_vertical_dimension_limits]
    end
    
    φ_maximum = maximum(φ_array)
    φ_minimum = minimum(φ_array)
    
    if use_symmetric_colorrange
        colorrange_limit = max(abs(φ_maximum), abs(φ_minimum))
        colorrange = (-colorrange_limit, colorrange_limit)
    else
        colorrange = (φ_minimum, φ_maximum)
    end
    
    return colorrange
end

function specify_colorrange_timeseries(grid, φ_series; use_symmetric_colorrange = true, ssh = false)
    Nx = grid.Nx
    Ny = grid.Ny
    Nz = grid.Nz
    
    n = length(φ_series)
    
    if ssh
        φ_series_array = zeros(Nx, Ny, 1, 6, n)
        φ_series_array_vertical_dimension_limits = Nz+1:Nz+1
    else
        φ_series_array = zeros(Nx, Ny, Nz, 6, n)
        φ_series_array_vertical_dimension_limits = 1:Nz
    end
    
    for i in 1:n
        for region in 1:6
            φ_series_array[:, :, :, region, i] = φ_series[i][region].data[1:Nx, 1:Ny, 
                                                                          φ_series_array_vertical_dimension_limits]
        end
    end
    
    φ_maximum = maximum(φ_series_array)
    φ_minimum = minimum(φ_series_array)
    
    if use_symmetric_colorrange
        colorrange_limit = max(abs(φ_maximum), abs(φ_minimum))
        colorrange = (-colorrange_limit, colorrange_limit)
    else
        colorrange = (φ_minimum, φ_maximum)
    end
    
    return colorrange
end

function panel_wise_visualization_of_grid_metrics_with_halos(metric; use_symmetric_colorrange = true)
    fig = Figure(resolution = (2450, 1400))

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, aspect = 1.0,
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 27.5, titlegap = 15, titlefont = :bold,
                   xlabel = "Local x direction", ylabel = "Local y direction")

    colorrange = specify_colorrange(metric; use_symmetric_colorrange = use_symmetric_colorrange)
    if use_symmetric_colorrange
        colormap = :balance
    else
        colormap = :amp
    end

    ax_1 = Axis(fig[3, 1]; title = "Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, metric[:, :, 1]; colorrange, colormap)
    Colorbar(fig[3, 2], hm_1)

    ax_2 = Axis(fig[3, 3]; title = "Panel 2", axis_kwargs...)
    hm_2 = heatmap!(ax_2, metric[:, :, 2]; colorrange, colormap)
    Colorbar(fig[3, 4], hm_2)

    ax_3 = Axis(fig[2, 3]; title = "Panel 3", axis_kwargs...)
    hm_3 = heatmap!(ax_3, metric[:, :, 3]; colorrange, colormap)
    Colorbar(fig[2, 4], hm_3)

    ax_4 = Axis(fig[2, 5]; title = "Panel 4", axis_kwargs...)
    hm_4 = heatmap!(ax_4, metric[:, :, 4]; colorrange, colormap)
    Colorbar(fig[2, 6], hm_4)

    ax_5 = Axis(fig[1, 5]; title = "Panel 5", axis_kwargs...)
    hm_5 = heatmap!(ax_5, metric[:, :, 5]; colorrange, colormap)
    Colorbar(fig[1, 6], hm_5)

    ax_6 = Axis(fig[1, 7]; title = "Panel 6", axis_kwargs...)
    hm_6 = heatmap!(ax_6, metric[:, :, 6]; colorrange, colormap)
    Colorbar(fig[1, 8], hm_6)

    return fig
end

function panel_wise_visualization_with_halos(grid, field; k = 1, use_symmetric_colorrange = true, ssh = false)
    fig = Figure(resolution = (2450, 1400))

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, aspect = 1.0, 
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 27.5, titlegap = 15, titlefont = :bold,
                   xlabel = "Local x direction", ylabel = "Local y direction")
    
    colorrange = specify_colorrange(grid, field; use_symmetric_colorrange = use_symmetric_colorrange, ssh = ssh)
    if use_symmetric_colorrange
        colormap = :balance
    else
        colormap = :amp
    end
    
    ax_1 = Axis(fig[3, 1]; title = "Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, parent(getregion(field, 1).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[3, 2], hm_1)

    ax_2 = Axis(fig[3, 3]; title = "Panel 2", axis_kwargs...)
    hm_2 = heatmap!(ax_2, parent(getregion(field, 2).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[3, 4], hm_2)

    ax_3 = Axis(fig[2, 3]; title = "Panel 3", axis_kwargs...)
    hm_3 = heatmap!(ax_3, parent(getregion(field, 3).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[2, 4], hm_3)

    ax_4 = Axis(fig[2, 5]; title = "Panel 4", axis_kwargs...)
    hm_4 = heatmap!(ax_4, parent(getregion(field, 4).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[2, 6], hm_4)

    ax_5 = Axis(fig[1, 5]; title = "Panel 5", axis_kwargs...)
    hm_5 = heatmap!(ax_5, parent(getregion(field, 5).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[1, 6], hm_5)

    ax_6 = Axis(fig[1, 7]; title = "Panel 6", axis_kwargs...)
    hm_6 = heatmap!(ax_6, parent(getregion(field, 6).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[1, 8], hm_6)

    return fig
end

function panel_wise_visualization(grid, field; k = 1, use_symmetric_colorrange = true, ssh = false)
    fig = Figure(resolution = (2450, 1400))
    
    Nx = grid.Nx
    Ny = grid.Ny

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, aspect = 1.0, 
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 27.5, titlegap = 15, titlefont = :bold,
                   xlabel = "Local x direction", ylabel = "Local y direction")
    
    colorrange = specify_colorrange(grid, field; use_symmetric_colorrange = use_symmetric_colorrange, ssh = ssh)
    if use_symmetric_colorrange
        colormap = :balance
    else
        colormap = :amp
    end
    
    ax_1 = Axis(fig[3, 1]; title = "Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, parent(getregion(field, 1).data[1:Nx, 1:Ny, k]); colorrange, colormap)
    Colorbar(fig[3, 2], hm_1)

    ax_2 = Axis(fig[3, 3]; title = "Panel 2", axis_kwargs...)
    hm_2 = heatmap!(ax_2, parent(getregion(field, 2).data[1:Nx, 1:Ny, k]); colorrange, colormap)
    Colorbar(fig[3, 4], hm_2)

    ax_3 = Axis(fig[2, 3]; title = "Panel 3", axis_kwargs...)
    hm_3 = heatmap!(ax_3, parent(getregion(field, 3).data[1:Nx, 1:Ny, k]); colorrange, colormap)
    Colorbar(fig[2, 4], hm_3)

    ax_4 = Axis(fig[2, 5]; title = "Panel 4", axis_kwargs...)
    hm_4 = heatmap!(ax_4, parent(getregion(field, 4).data[1:Nx, 1:Ny, k]); colorrange, colormap)
    Colorbar(fig[2, 6], hm_4)

    ax_5 = Axis(fig[1, 5]; title = "Panel 5", axis_kwargs...)
    hm_5 = heatmap!(ax_5, parent(getregion(field, 5).data[1:Nx, 1:Ny, k]); colorrange, colormap)
    Colorbar(fig[1, 6], hm_5)

    ax_6 = Axis(fig[1, 7]; title = "Panel 6", axis_kwargs...)
    hm_6 = heatmap!(ax_6, parent(getregion(field, 6).data[1:Nx, 1:Ny, k]); colorrange, colormap)
    Colorbar(fig[1, 8], hm_6)

    return fig
end

function geo_heatlatlon_visualization(grid, field, title; k = 1, use_symmetric_colorrange = true, ssh = false,
                                      cbar_label = "", specify_plot_limits = false, plot_limits = [])
    fig = Figure(resolution = (1350, 650))

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5,
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 25, titlegap = 15, titlefont = :bold)

    if specify_plot_limits
        colorrange = plot_limits
    else
        colorrange = specify_colorrange(grid, field; use_symmetric_colorrange = use_symmetric_colorrange, ssh = ssh)
    end

    if use_symmetric_colorrange
        colormap = :balance
    else
        colormap = :amp
    end

    ax = GeoAxis(fig[1, 1]; coastlines = true, lonlims = automatic, title = title, axis_kwargs...)
    heatlatlon!(ax, field, k; colorrange, colormap)

    Colorbar(fig[1, 2], limits = colorrange, colormap = colormap, label = cbar_label, labelsize = 22.5,
             labelpadding = 10, ticksize = 17.5, width = 25, height = Relative(0.9))

    colsize!(fig.layout, 1, Auto(0.8))
    colgap!(fig.layout, 75)

    return fig
end

function geo_heatlatlon_visualization_animation(grid, fields, location, prettytimes, title_prefix; start_index = 1,
                                                k = 1, use_symmetric_colorrange = true, ssh = false, cbar_label = "",
                                                specify_plot_limits = false, plot_limits = [], framerate = 10,
                                                filename = "animation")
    n = Observable(start_index)
    
    fig = Figure(resolution=(1350, 650))

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5,
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 25, titlegap = 15, titlefont = :bold)

    if specify_plot_limits
        colorrange = plot_limits
    else
        colorrange = specify_colorrange_timeseries(grid, fields; use_symmetric_colorrange = use_symmetric_colorrange,
                                                   ssh = ssh)
    end

    if use_symmetric_colorrange
        colormap = :balance
    else
        colormap = :amp
    end
    
    field = @lift fields[$n]
    prettytime = @lift prettytimes[$n]

    ax = GeoAxis(fig[1, 1]; coastlines = true, lonlims = automatic, axis_kwargs...)
    ax.title = title_prefix * " after " * prettytime[]
    interpolated_field = interpolate_cubed_sphere_field_to_cell_centers(grid, field[], location; k = k)
    heatlatlon!(ax, interpolated_field, k; colorrange, colormap)
    Colorbar(fig[1, 2], limits = colorrange, colormap = colormap, label = cbar_label, labelsize = 22.5,
             labelpadding = 10, ticksize = 17.5, width = 25, height = Relative(0.9))
    colsize!(fig.layout, 1, Auto(0.8))
    colgap!(fig.layout, 75)
    
    frames = 1:length(fields)

    CairoMakie.record(fig, filename * ".mp4", frames, framerate = framerate) do i
        msg = string("Plotting frame ", i, " of ", frames[end])
        print(msg * " \r")

        field[] = fields[i]
        prettytime[] = prettytimes[i]

        # Update the title of the plot
        ax.title = title_prefix * " after " * prettytime[]

        # Update the plot
        interpolated_field = interpolate_cubed_sphere_field_to_cell_centers(grid, field[], location; k = k)
        heatlatlon!(ax, interpolated_field, k; colorrange, colormap)
        Colorbar(fig[1, 2], limits = colorrange, colormap = colormap, label = cbar_label, labelsize = 22.5,
                 labelpadding = 10, ticksize = 17.5, width = 25, height = Relative(0.9))
        colsize!(fig.layout, 1, Auto(0.8))
        colgap!(fig.layout, 75)
    end
end

function create_heat_map_or_contour_plot_animation(plot_type, x, y, φ_series, resolution, axis_kwargs, contourlevels,
                                                   cbar_kwargs, framerate, filename; start_index = 1,
                                                   use_symmetric_colorrange = true)

    n = Observable(start_index)
    φ = @lift begin
        φ_series[:, :, $n]
    end

    fig = Figure(resolution = resolution)
    # Specify the title of every frame if desired.
    ax = Axis(fig[1,1]; axis_kwargs...)
    colorrange = specify_colorrange(φ_series; use_symmetric_colorrange = use_symmetric_colorrange)
    if use_symmetric_colorrange
        colormap = :balance
    else
        colormap = :amp
    end
    
    if plot_type == "filled_contour_plot"
        myplot = contourf!(ax, x, y, φ; levels = range(colorrange..., length=contourlevels), colormap = colormap)  
    elseif plot_type == "heat_map"
        myplot = heatmap!(ax, x, y, φ; colorrange = colorrange, colormap = colormap)
    end
    Colorbar(fig[1,2], myplot; cbar_kwargs...)
    
    frames = 1:size(φ_series, 3)
    CairoMakie.record(fig, filename * ".mp4", frames, framerate = framerate) do i
        msg = string("Plotting frame ", i, " of ", frames[end])
        print(msg * " \r")
        n[] = i
    end

end

function test_create_heat_map_or_contour_plot_animation()
    nPoints = 50
    x = range(0, 2π, length = nPoints)
    y = range(0, 2π, length = nPoints)
    t = range(0, 2π, length = nPoints)
    k = 1
    l = 1
    ω = 1
    φ_series = zeros(nPoints, nPoints, nPoints)
    for i in 1:nPoints
        for j in 1:nPoints
            for m in 1:nPoints
                φ_series[i, j, m] = sin(k * x[i] + l * y[j] - ω * t[m])
            end
        end
    end
    axis_kwargs = (xlabel = "x", ylabel = "y", xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, 
                   yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1, 
                   title = "sin(kx + ly - ωt)", titlesize = 27.5, titlegap = 15, titlefont = :bold)
    contourlevels = 50
    cbar_kwargs = (label = "sin(kx + ly - ωt)", labelsize = 22.5, labelpadding = 10, ticksize = 17.5)
    framerate = 10
    create_heat_map_or_contour_plot_animation("heat_map", x, y, φ_series, (850, 750), axis_kwargs, contourlevels,
                                              cbar_kwargs, framerate, "sine_wave_heat_map")
    create_heat_map_or_contour_plot_animation("filled_contour_plot", x, y, φ_series, (850, 750), axis_kwargs, 
                                              contourlevels, cbar_kwargs, framerate, "sine_wave_filled_contour_plot")
end

function create_panel_wise_visualization_animation_with_halos(grid, φ_series, framerate, filename; start_index = 1,
                                                              k = 1, use_symmetric_colorrange = true, ssh = false)
    n = Observable(start_index) # the current index

    colorrange = specify_colorrange(grid, φ_series; use_symmetric_colorrange = use_symmetric_colorrange, ssh = ssh)
    if use_symmetric_colorrange
        colormap = :balance
    else
        colormap = :amp
    end

    # Create the initial visualization.
    fig = Figure(resolution = (2450, 1400))
    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, aspect = 1.0, 
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 27.5, titlegap = 15, titlefont = :bold,
                   xlabel = "Local x direction", ylabel = "Local y direction")
    
    φ = @lift φ_series[$n]

    ax_1 = Axis(fig[3, 1]; title = "Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, parent(φ[][1].data[:, :, k]); colorrange, colormap)
    Colorbar(fig[3, 2], hm_1)

    ax_2 = Axis(fig[3, 3]; title = "Panel 2", axis_kwargs...)
    hm_2 = heatmap!(ax_2, parent(φ[][2].data[:, :, k]); colorrange, colormap)
    Colorbar(fig[3, 4], hm_2)

    ax_3 = Axis(fig[2, 3]; title = "Panel 3", axis_kwargs...)
    hm_3 = heatmap!(ax_3, parent(φ[][3].data[:, :, k]); colorrange, colormap)
    Colorbar(fig[2, 4], hm_3)

    ax_4 = Axis(fig[2, 5]; title = "Panel 4", axis_kwargs...)
    hm_4 = heatmap!(ax_4, parent(φ[][4].data[:, :, k]); colorrange, colormap)
    Colorbar(fig[2, 6], hm_4)

    ax_5 = Axis(fig[1, 5]; title = "Panel 5", axis_kwargs...)
    hm_5 = heatmap!(ax_5, parent(φ[][5].data[:, :, k]); colorrange, colormap)
    Colorbar(fig[1, 6], hm_5)

    ax_6 = Axis(fig[1, 7]; title = "Panel 6", axis_kwargs...)
    hm_6 = heatmap!(ax_6, parent(φ[][6].data[:, :, k]); colorrange, colormap)
    Colorbar(fig[1, 8], hm_6)

    frames = 1:length(φ_series)

    CairoMakie.record(fig, filename * ".mp4", frames, framerate = framerate) do i
        msg = string("Plotting frame ", i, " of ", frames[end])
        print(msg * " \r")

        φ[] = φ_series[i]

        hm_1 = heatmap!(ax_1, parent(φ[][1].data[:, :, k]); colorrange, colormap)
        Colorbar(fig[3, 2], hm_1)

        hm_2 = heatmap!(ax_2, parent(φ[][2].data[:, :, k]); colorrange, colormap)
        Colorbar(fig[3, 4], hm_2)

        hm_3 = heatmap!(ax_3, parent(φ[][3].data[:, :, k]); colorrange, colormap)
        Colorbar(fig[2, 4], hm_3)

        hm_4 = heatmap!(ax_4, parent(φ[][4].data[:, :, k]); colorrange, colormap)
        Colorbar(fig[2, 6], hm_4)

        hm_5 = heatmap!(ax_5, parent(φ[][5].data[:, :, k]); colorrange, colormap)
        Colorbar(fig[1, 6], hm_5)

        hm_6 = heatmap!(ax_6, parent(φ[][6].data[:, :, k]); colorrange, colormap)
        Colorbar(fig[1, 8], hm_6)
    end

end

function create_panel_wise_visualization_animation(grid, φ_series, framerate, filename; start_index = 1, k = 1,
                                                   use_symmetric_colorrange = true, ssh = false)
    Nx = grid.Nx
    Ny = grid.Ny

    n = Observable(start_index) # the current index

    colorrange = specify_colorrange_timeseries(grid, φ_series; use_symmetric_colorrange=use_symmetric_colorrange,
                                               ssh = ssh)
    if use_symmetric_colorrange
        colormap = :balance
    else
        colormap = :amp
    end

    # Create the initial visualization.
    fig = Figure(resolution = (2450, 1400))
    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, aspect = 1.0, 
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 27.5, titlegap = 15, titlefont = :bold,
                   xlabel = "Local x direction", ylabel = "Local y direction")

    φ = @lift φ_series[$n]

    ax_1 = Axis(fig[3, 1]; title="Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, φ[][1].data[1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[3, 2], hm_1)

    ax_2 = Axis(fig[3, 3]; title="Panel 2", axis_kwargs...)
    hm_2 = heatmap!(ax_2, φ[][2].data[1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[3, 4], hm_2)

    ax_3 = Axis(fig[2, 3]; title="Panel 3", axis_kwargs...)
    hm_3 = heatmap!(ax_3, φ[][3].data[1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[2, 4], hm_3)

    ax_4 = Axis(fig[2, 5]; title="Panel 4", axis_kwargs...)
    hm_4 = heatmap!(ax_4, φ[][4].data[1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[2, 6], hm_4)

    ax_5 = Axis(fig[1, 5]; title="Panel 5", axis_kwargs...)
    hm_5 = heatmap!(ax_5, φ[][5].data[1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[1, 6], hm_5)

    ax_6 = Axis(fig[1, 7]; title="Panel 6", axis_kwargs...)
    hm_6 = heatmap!(ax_6, φ[][6].data[1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[1, 8], hm_6)

    frames = 1:length(φ_series)

    CairoMakie.record(fig, filename * ".mp4", frames, framerate = framerate) do i
        msg = string("Plotting frame ", i, " of ", frames[end])
        print(msg * " \r")

        φ[] = φ_series[i]

        hm_1 = heatmap!(ax_1, φ[][1].data[1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[3, 2], hm_1)

        hm_2 = heatmap!(ax_2, φ[][2].data[1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[3, 4], hm_2)

        hm_3 = heatmap!(ax_3, φ[][3].data[1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[2, 4], hm_3)

        hm_4 = heatmap!(ax_4, φ[][4].data[1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[2, 6], hm_4)

        hm_5 = heatmap!(ax_5, φ[][5].data[1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[1, 6], hm_5)

        hm_6 = heatmap!(ax_6, φ[][6].data[1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[1, 8], hm_6)
    end
end
