using LinearAlgebra
using Oceananigans.MultiRegion: getregion
using CairoMakie

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
                                       titlegap, colormap, contourlevels, specify_axis_limits = true, 
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

function read_big_endian_coordinates(filename)
    # Open the file in binary read mode
    open(filename, "r") do io
        # Calculate the number of Float64 values in the file
        n = filesize(io) ÷ sizeof(Float64)
        
        # Ensure n = 32x32
        if n != 32 * 32
            error("File size does not match the expected size for one 32x32 field")
        end

        # Initialize an array to hold the data
        data = Vector{Float64}(undef, n)

        # Read the data into the array
        read!(io, data)

        # Convert from big-endian to native endianness
        native_data = reshape(bswap.(data), 32, 32)

        return native_data
    end
end

function read_big_endian_diagnostic_data(filename)
    # Open the file in binary read mode
    open(filename, "r") do io
        # Calculate the number of Float64 values in the file
        n = filesize(io) ÷ sizeof(Float64)

        # Ensure n = 2x32x32
        if n != 2 * 32 * 32
            error("File size does not match the expected size for two 32x32 fields")
        end

        # Initialize an array to hold the data
        data = Vector{Float64}(undef, n)

        # Read the data into the array
        read!(io, data)

        # Convert from big-endian to native endianness
        native_data = bswap.(data)

        # Extract and reshape the data to form two 32x32 fields
        momKE = reshape(native_data[1:32*32], 32, 32)
        momVort3 = reshape(native_data[32*32+1:end], 32, 32)

        return momKE, momVort3
    end
end

function specify_colorrange_MITgcm(φ, use_symmetric_colorrange = true)

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

function specify_colorrange(grid, φ, use_symmetric_colorrange = true)

    Nx = grid.Nx
    Ny = grid.Ny
    Nz = grid.Nz
    
    φ_array = zeros(Nx, Ny, Nz, 6)
    
    for region in 1:6
        φ_array[:, :, :, region] = φ[region].data[1:Nx, 1:Ny, 1:Nz]
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

function specify_colorrange_timeseries(grid, φ_series, use_symmetric_colorrange = true)

    Nx = grid.Nx
    Ny = grid.Ny
    Nz = grid.Nz
    
    n = size(φ_series)[1]
    φ_series_array = zeros(Nx, Ny, Nz, n)
    
    for i in 1:n
        for region in 1:6
            φ_series_array[:, :, :, i] = φ_series[i][region].data[1:Nx, 1:Ny, 1:Nz]
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

function panel_wise_visualization_MITgcm(x, y, field, use_symmetric_colorrange)

    fig = Figure(resolution = (2450, 1400))

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, aspect = 1.0, 
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 27.5, titlegap = 15, titlefont = :bold,
                   xlabel = "Local x direction", ylabel = "Local y direction")
    
    colorrange = specify_colorrange_MITgcm(field, use_symmetric_colorrange)
    if use_symmetric_colorrange
        colormap = :balance
    else
        colormap = :amp
    end
    
    ax_1 = Axis(fig[3, 1]; title = "Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, x[:, :, 1], y[:, :, 1], field[:, :, 1]; colorrange, colormap)
    Colorbar(fig[3, 2], hm_1)

    ax_2 = Axis(fig[3, 3]; title = "Panel 2", axis_kwargs...)
    hm_2 = heatmap!(ax_2, x[:, :, 2], y[:, :, 2], field[:, :, 2]; colorrange, colormap)
    Colorbar(fig[3, 4], hm_2)

    ax_3 = Axis(fig[2, 3]; title = "Panel 3", axis_kwargs...)
    hm_3 = heatmap!(ax_3, x[:, :, 3], y[:, :, 3], field[:, :, 3]; colorrange, colormap)
    Colorbar(fig[2, 4], hm_3)

    ax_4 = Axis(fig[2, 5]; title = "Panel 4", axis_kwargs...)
    hm_4 = heatmap!(ax_4, x[:, :, 4], y[:, :, 4], field[:, :, 4]; colorrange, colormap)
    Colorbar(fig[2, 6], hm_4)

    ax_5 = Axis(fig[1, 5]; title = "Panel 5", axis_kwargs...)
    hm_5 = heatmap!(ax_5, x[:, :, 5], y[:, :, 5], field[:, :, 5]; colorrange, colormap)
    Colorbar(fig[1, 6], hm_5)

    ax_6 = Axis(fig[1, 7]; title = "Panel 6", axis_kwargs...)
    hm_6 = heatmap!(ax_6, x[:, :, 6], y[:, :, 6], field[:, :, 6]; colorrange, colormap)
    Colorbar(fig[1, 8], hm_6)

    return fig
    
end

function panel_wise_visualization_with_halos(grid, field, k = 1, use_symmetric_colorrange = true)
    fig = Figure(resolution = (2450, 1400))

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, aspect = 1.0, 
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 27.5, titlegap = 15, titlefont = :bold,
                   xlabel = "Local x direction", ylabel = "Local y direction")
    
    colorrange = specify_colorrange(grid, field, use_symmetric_colorrange)
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

function panel_wise_visualization(grid, field, k = 1, use_symmetric_colorrange = true)
    fig = Figure(resolution = (2450, 1400))
    
    Nx = grid.Nx
    Ny = grid.Ny

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, aspect = 1.0, 
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 27.5, titlegap = 15, titlefont = :bold,
                   xlabel = "Local x direction", ylabel = "Local y direction")
    
    colorrange = specify_colorrange(grid, field, use_symmetric_colorrange)
    if use_symmetric_colorrange
        colormap = :balance
    else
        colormap = :amp
    end
    
    ax_1 = Axis(fig[3, 1]; title = "Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, getregion(field, 1).data[1:Nx, 1:Ny, k]; colorrange, colormap)
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

function create_heat_map_or_contour_plot_animation(plot_type, x, y, φ_series, start_index, resolution, axis_kwargs, 
                                                   use_symmetric_colorrange, contourlevels, cbar_kwargs, framerate, 
                                                   filename)

    n = Observable(start_index)
    φ = @lift begin
        φ_series[:, :, $n]
    end

    fig = Figure(resolution = resolution)
    # Specify the title of every frame if desired.
    ax = Axis(fig[1,1]; axis_kwargs...)
    colorrange = specify_colorrange_MITgcm(φ_series, use_symmetric_colorrange)
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
    use_symmetric_colorrange = true
    contourlevels = 50
    cbar_kwargs = (label = "sin(kx + ly - ωt)", labelsize = 22.5, labelpadding = 10, ticksize = 17.5)
    framerate = 10
    create_heat_map_or_contour_plot_animation("heat_map", x, y, φ_series, 1, (850, 750), axis_kwargs, 
                                              use_symmetric_colorrange, contourlevels, cbar_kwargs, framerate, 
                                              "sine_wave_heat_map")
    create_heat_map_or_contour_plot_animation("filled_contour_plot", x, y, φ_series, 1, (850, 750), axis_kwargs, 
                                              use_symmetric_colorrange, contourlevels, cbar_kwargs, framerate, 
                                              "sine_wave_filled_contour_plot")
end

function create_panel_wise_visualization_animation_MITgcm(x, y, φ_series, start_index, use_symmetric_colorrange, 
                                                          framerate, filename)

    n = Observable(start_index) # the current index

    colorrange = specify_colorrange_MITgcm(φ_series, use_symmetric_colorrange)
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
    
    ax_1 = Axis(fig[3, 1]; title = "Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, x[:, :, 1], y[:, :, 1], φ_series[:, :, 1, start_index]; colorrange, colormap)
    Colorbar(fig[3, 2], hm_1)

    ax_2 = Axis(fig[3, 3]; title = "Panel 2", axis_kwargs...)
    hm_2 = heatmap!(ax_2, x[:, :, 2], y[:, :, 2], φ_series[:, :, 2, start_index]; colorrange, colormap)
    Colorbar(fig[3, 4], hm_2)

    ax_3 = Axis(fig[2, 3]; title = "Panel 3", axis_kwargs...)
    hm_3 = heatmap!(ax_3, x[:, :, 3], y[:, :, 3], φ_series[:, :, 3, start_index]; colorrange, colormap)
    Colorbar(fig[2, 4], hm_3)

    ax_4 = Axis(fig[2, 5]; title = "Panel 4", axis_kwargs...)
    hm_4 = heatmap!(ax_4, x[:, :, 4], y[:, :, 4], φ_series[:, :, 4, start_index]; colorrange, colormap)
    Colorbar(fig[2, 6], hm_4)

    ax_5 = Axis(fig[1, 5]; title = "Panel 5", axis_kwargs...)
    hm_5 = heatmap!(ax_5, x[:, :, 5], y[:, :, 5], φ_series[:, :, 5, start_index]; colorrange, colormap)
    Colorbar(fig[1, 6], hm_5)

    ax_6 = Axis(fig[1, 7]; title = "Panel 6", axis_kwargs...)
    hm_6 = heatmap!(ax_6, x[:, :, 6], y[:, :, 6], φ_series[:, :, 6, start_index]; colorrange, colormap)
    Colorbar(fig[1, 8], hm_6)
    
    # Use an on block to reactively update the visualization.
    on(n) do index
        hm_1 = heatmap!(ax_1, x[:, :, 1], y[:, :, 1], φ_series[:, :, 1, index]; colorrange, colormap)
        Colorbar(fig[3, 2], hm_1)

        hm_2 = heatmap!(ax_2, x[:, :, 2], y[:, :, 2], φ_series[:, :, 2, index]; colorrange, colormap)
        Colorbar(fig[3, 4], hm_2)

        hm_3 = heatmap!(ax_3, x[:, :, 3], y[:, :, 3], φ_series[:, :, 3, index]; colorrange, colormap)
        Colorbar(fig[2, 4], hm_3)

        hm_4 = heatmap!(ax_4, x[:, :, 4], y[:, :, 4], φ_series[:, :, 4, index]; colorrange, colormap)
        Colorbar(fig[2, 6], hm_4)

        hm_5 = heatmap!(ax_5, x[:, :, 5], y[:, :, 5], φ_series[:, :, 5, index]; colorrange, colormap)
        Colorbar(fig[1, 6], hm_5)

        hm_6 = heatmap!(ax_6, x[:, :, 6], y[:, :, 6], φ_series[:, :, 6, index]; colorrange, colormap)
        Colorbar(fig[1, 8], hm_6)
    end
    
    frames = 1:size(φ_series, 4)
    
    CairoMakie.record(fig, filename * ".mp4", frames, framerate = framerate) do i
        msg = string("Plotting frame ", i, " of ", frames[end])
        print(msg * " \r")
        n[] = i
    end

end

function create_panel_wise_visualization_animation_with_halos(grid, φ_series, start_index, use_symmetric_colorrange, 
                                                              framerate, filename, k=1)

    n = Observable(start_index) # the current index

    colorrange = specify_colorrange(grid, φ_series, use_symmetric_colorrange)
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

    ax_1 = Axis(fig[3, 1]; title = "Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, parent(φ_series[start_index][1].data[:, :, k]); colorrange, colormap)
    Colorbar(fig[3, 2], hm_1)

    ax_2 = Axis(fig[3, 3]; title = "Panel 2", axis_kwargs...)
    hm_2 = heatmap!(ax_2, parent(φ_series[start_index][2].data[:, :, k]); colorrange, colormap)
    Colorbar(fig[3, 4], hm_2)

    ax_3 = Axis(fig[2, 3]; title = "Panel 3", axis_kwargs...)
    hm_3 = heatmap!(ax_3, parent(φ_series[start_index][3].data[:, :, k]); colorrange, colormap)
    Colorbar(fig[2, 4], hm_3)

    ax_4 = Axis(fig[2, 5]; title = "Panel 4", axis_kwargs...)
    hm_4 = heatmap!(ax_4, parent(φ_series[start_index][4].data[:, :, k]); colorrange, colormap)
    Colorbar(fig[2, 6], hm_4)

    ax_5 = Axis(fig[1, 5]; title = "Panel 5", axis_kwargs...)
    hm_5 = heatmap!(ax_5, parent(φ_series[start_index][5].data[:, :, k]); colorrange, colormap)
    Colorbar(fig[1, 6], hm_5)

    ax_6 = Axis(fig[1, 7]; title = "Panel 6", axis_kwargs...)
    hm_6 = heatmap!(ax_6, parent(φ_series[start_index][6].data[:, :, k]); colorrange, colormap)
    Colorbar(fig[1, 8], hm_6)

    # Use an on block to reactively update the visualization.
    on(n) do index
        hm_1 = heatmap!(ax_1, parent(φ_series[index][1].data[:, :, k]); colorrange, colormap)
        Colorbar(fig[3, 2], hm_1)

        hm_2 = heatmap!(ax_2, parent(φ_series[index][2].data[:, :, k]); colorrange, colormap)
        Colorbar(fig[3, 4], hm_2)

        hm_3 = heatmap!(ax_3, parent(φ_series[index][3].data[:, :, k]); colorrange, colormap)
        Colorbar(fig[2, 4], hm_3)

        hm_4 = heatmap!(ax_4, parent(φ_series[index][4].data[:, :, k]); colorrange, colormap)
        Colorbar(fig[2, 6], hm_4)

        hm_5 = heatmap!(ax_5, parent(φ_series[index][5].data[:, :, k]); colorrange, colormap)
        Colorbar(fig[1, 6], hm_5)

        hm_6 = heatmap!(ax_6, parent(φ_series[index][6].data[:, :, k]); colorrange, colormap)
        Colorbar(fig[1, 8], hm_6)
    end

    frames = 1:size(φ_series)[1]

    CairoMakie.record(fig, filename * ".mp4", frames, framerate = framerate) do i
        msg = string("Plotting frame ", i, " of ", frames[end])
        print(msg * " \r")
        n[] = i
    end

end

function create_panel_wise_visualization_animation(grid, φ_series, start_index, use_symmetric_colorrange, 
                                                   framerate, filename, k=1)
    
    Nx = grid.Nx
    Ny = grid.Ny
    
    n = Observable(start_index) # the current index

    colorrange = specify_colorrange_timeseries(grid, φ_series, use_symmetric_colorrange)
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

    ax_1 = Axis(fig[3, 1]; title = "Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, φ_series[start_index][1].data[1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[3, 2], hm_1)

    ax_2 = Axis(fig[3, 3]; title = "Panel 2", axis_kwargs...)
    hm_2 = heatmap!(ax_2, φ_series[start_index][2].data[1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[3, 4], hm_2)

    ax_3 = Axis(fig[2, 3]; title = "Panel 3", axis_kwargs...)
    hm_3 = heatmap!(ax_3, φ_series[start_index][3].data[1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[2, 4], hm_3)

    ax_4 = Axis(fig[2, 5]; title = "Panel 4", axis_kwargs...)
    hm_4 = heatmap!(ax_4, φ_series[start_index][4].data[1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[2, 6], hm_4)

    ax_5 = Axis(fig[1, 5]; title = "Panel 5", axis_kwargs...)
    hm_5 = heatmap!(ax_5, φ_series[start_index][5].data[1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[1, 6], hm_5)

    ax_6 = Axis(fig[1, 7]; title = "Panel 6", axis_kwargs...)
    hm_6 = heatmap!(ax_6, φ_series[start_index][6].data[1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[1, 8], hm_6)

    # Use an on block to reactively update the visualization.
    on(n) do index
        hm_1 = heatmap!(ax_1, φ_series[index][1].data[1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[3, 2], hm_1)

        hm_2 = heatmap!(ax_2, φ_series[index][2].data[1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[3, 4], hm_2)

        hm_3 = heatmap!(ax_3, φ_series[index][3].data[1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[2, 4], hm_3)

        hm_4 = heatmap!(ax_4, φ_series[index][4].data[1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[2, 6], hm_4)

        hm_5 = heatmap!(ax_5, φ_series[index][5].data[1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[1, 6], hm_5)

        hm_6 = heatmap!(ax_6, φ_series[index][6].data[1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[1, 8], hm_6)
    end

    frames = 1:size(φ_series)[1]

    CairoMakie.record(fig, filename * ".mp4", frames, framerate = framerate) do i
        msg = string("Plotting frame ", i, " of ", frames[end])
        print(msg * " \r")
        n[] = i
    end

end