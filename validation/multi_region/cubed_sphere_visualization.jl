using LinearAlgebra
using Printf
using Oceananigans.MultiRegion: getregion
using CairoMakie, GeoMakie, Imaginocean

function interpolate_cubed_sphere_field_to_cell_centers(grid, field, location; ssh = false, levels = 1:1,
                                                        read_parent_field_data = false)
    if location == "cc" && !read_parent_field_data
        return field
    end
    
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    hx = read_parent_field_data ? Hx : 0
    hy = read_parent_field_data ? Hy : 0
    hz = read_parent_field_data ? Hz : 0

    if ssh
        interpolated_field = Field{Center, Center, Center}(grid, indices = (:, :, Nz+1:Nz+1))
        levels_field = read_parent_field_data ? (1:1) : (Nz+1:Nz+1)
        levels_interpolated_field = Nz+1:Nz+1

    else
        interpolated_field = Field{Center, Center, Center}(grid)
        levels_field = levels .+ hz
        levels_interpolated_field = levels
    end
    set!(interpolated_field, 0)

    for region in 1:number_of_regions(grid), j in 1:Ny, i in 1:Nx
        if location == "fc"
            interpolated_field[region][i, j, levels_interpolated_field] = (
            0.5(field[region][i+hx, j+hy, levels_field] + field[region][i+1+hx, j+hy, levels_field]))
        elseif location == "cf"
            interpolated_field[region][i, j, levels_interpolated_field] = (
            0.5(field[region][i+hx, j+hy, levels_field] + field[region][i+hx, j+1+hy, levels_field]))
        elseif location == "ff"
            interpolated_field[region][i, j, levels_interpolated_field] = (
            0.25(field[region][i+hx, j+hy, levels_field] + field[region][i+1+hx, j+hy, levels_field]
                 + field[region][i+1+hx, j+1+hy, levels_field] + field[region][i+hx, j+1+hy, levels_field]))
        elseif location == "cc" && read_parent_field_data
            interpolated_field[region][i, j, levels_interpolated_field] = field[region][i+hx, j+hy, levels_field]
        end
    end

    return interpolated_field
end

function calculate_sines_and_cosines_of_cubed_sphere_grid_angles(grid, location)
    Nx, Ny, Nz = size(grid)

    cos_θ = zeros(Nx, Ny, number_of_regions(grid))
    sin_θ = zeros(Nx, Ny, number_of_regions(grid))

    for region in 1:number_of_regions(grid), j in 1:Ny, i in 1:Nx
        if location == "cc"
            u_Pseudo = deg2rad(grid[region].φᶜᶠᵃ[i, j+1] - grid[region].φᶜᶠᵃ[i, j])/grid[region].Δyᶜᶜᵃ[i, j]
            v_Pseudo = -deg2rad(grid[region].φᶠᶜᵃ[i+1, j] - grid[region].φᶠᶜᵃ[i, j])/grid[region].Δxᶜᶜᵃ[i, j]
        elseif location == "fc"
            u_Pseudo = deg2rad(grid[region].φᶠᶠᵃ[i, j+1] - grid[region].φᶠᶠᵃ[i, j])/grid[region].Δyᶠᶜᵃ[i, j]
            v_Pseudo = -deg2rad(grid[region].φᶜᶜᵃ[i, j] - grid[region].φᶜᶜᵃ[i-1, j])/grid[region].Δxᶠᶜᵃ[i, j]
        elseif location == "cf"
            u_Pseudo = deg2rad(grid[region].φᶜᶜᵃ[i, j] - grid[region].φᶜᶜᵃ[i, j-1])/grid[region].Δyᶜᶠᵃ[i, j]
            v_Pseudo = -deg2rad(grid[region].φᶠᶠᵃ[i+1, j] - grid[region].φᶠᶠᵃ[i, j])/grid[region].Δxᶜᶠᵃ[i, j]
        end
        cos_θ[i, j, region] = u_Pseudo/sqrt(u_Pseudo^2 + v_Pseudo^2)
        sin_θ[i, j, region] = v_Pseudo/sqrt(u_Pseudo^2 + v_Pseudo^2)
    end

    return cos_θ, sin_θ
end

function orient_in_global_direction!(grid, u, v, cos_θ, sin_θ; levels = 1:1, read_parent_field_data = false)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    hx = read_parent_field_data ? Hx : 0
    hy = read_parent_field_data ? Hy : 0
    hz = read_parent_field_data ? Hz : 0

    for region in 1:number_of_regions(grid), k in levels, j in 1:Ny, i in 1:Nx
        uAtPoint = u[region][i+hx, j+hy, k+hz]
        vAtPoint = v[region][i+hx, j+hy, k+hz]
        u[region][i, j, k] = uAtPoint * cos_θ[i, j, region] + vAtPoint * sin_θ[i, j, region]
        v[region][i, j, k] = vAtPoint * cos_θ[i, j, region] - uAtPoint * sin_θ[i, j, region]
    end
end

function orient_in_local_direction!(grid, u, v, cos_θ, sin_θ; levels = 1:1, read_parent_field_data = false)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    hx = read_parent_field_data ? Hx : 0
    hy = read_parent_field_data ? Hy : 0
    hz = read_parent_field_data ? Hz : 0

    for region in 1:number_of_regions(grid), k in levels, j in 1:Ny, i in 1:Nx
        uAtPoint = u[region][i+hx, j+hy, k+hz]
        vAtPoint = v[region][i+hx, j+hy, k+hz]
        u[region][i, j, k] = uAtPoint * cos_θ[i, j, region] - vAtPoint * sin_θ[i, j, region]
        v[region][i, j, k] = vAtPoint * cos_θ[i, j, region] + uAtPoint * sin_θ[i, j, region]
    end
end

function orient_velocities_in_global_direction(grid, u, v, cos_θ, sin_θ; levels = 1:1, read_parent_field_data = false)
    u = interpolate_cubed_sphere_field_to_cell_centers(grid, u, "fc"; levels = levels,
                                                       read_parent_field_data = read_parent_field_data)
    v = interpolate_cubed_sphere_field_to_cell_centers(grid, v, "cf"; levels = levels,
                                                       read_parent_field_data = read_parent_field_data)
    orient_in_global_direction!(grid, u, v, cos_θ, sin_θ; levels = levels)
    return u, v
end

function extract_latitude(grid; location = "cc")
    Nx, Ny, Nz = size(grid)

    latitude = zeros(Nx, Ny, number_of_regions(grid))

    for region in 1:number_of_regions(grid), j in 1:Ny, i in 1:Nx
        if location == "cc"
            latitude[i, j, region] = grid[region].φᶜᶜᵃ[i, j]
        elseif location == "fc"
            latitude[i, j, region] = grid[region].φᶠᶜᵃ[i, j]
        elseif location == "cf"
            latitude[i, j, region] = grid[region].φᶜᶠᵃ[i, j]
        elseif location == "ff"
            latitude[i, j, region] = grid[region].φᶠᶠᵃ[i, j]
        end
    end

    return latitude
end

function extract_longitude(grid; location = "cc")
    Nx, Ny, Nz = size(grid)

    longitude = zeros(Nx, Ny, number_of_regions(grid))

    for region in 1:number_of_regions(grid), j in 1:Ny, i in 1:Nx
        if location == "cc"
            longitude[i, j, region] = grid[region].λᶜᶜᵃ[i, j]
        elseif location == "fc"
            longitude[i, j, region] = grid[region].λᶠᶜᵃ[i, j]
        elseif location == "cf"
            longitude[i, j, region] = grid[region].λᶜᶠᵃ[i, j]
        elseif location == "ff"
            longitude[i, j, region] = grid[region].λᶠᶠᵃ[i, j]
        end
    end

    return longitude
end

function extract_scalar_at_specific_longitude_through_panel_center(grid, scalar, panel_index)
    Nc = Nx
    east_panel_index = grid.connectivity.connections[panel_index].east.from_rank
    west_panel_index = grid.connectivity.connections[panel_index].west.from_rank
    north_panel_index = grid.connectivity.connections[panel_index].north.from_rank
    south_panel_index = grid.connectivity.connections[panel_index].south.from_rank
    scalar_at_specific_longitude_through_panel_center = zeros(2*Nc)
    NcBy2 = round(Int, Nc/2)
    if panel_index == 1
        scalar_at_specific_longitude_through_panel_center[1:NcBy2] = scalar[NcBy2, NcBy2+1:Nc, south_panel_index]
        scalar_at_specific_longitude_through_panel_center[NcBy2+1:round(Int, 3Nc/2)] = scalar[NcBy2, 1:Nc, panel_index]
        scalar_at_specific_longitude_through_panel_center[round(Int, 3Nc/2)+1:2*Nc] = (
        scalar[1:NcBy2, NcBy2+1, north_panel_index])
    elseif panel_index == 2
        scalar_at_specific_longitude_through_panel_center[1:NcBy2] = scalar[NcBy2+1:Nc, NcBy2+1, south_panel_index]
        scalar_at_specific_longitude_through_panel_center[NcBy2+1:round(Int, 3Nc/2)] = scalar[NcBy2, 1:Nc, panel_index]
        scalar_at_specific_longitude_through_panel_center[round(Int, 3Nc/2)+1:2*Nc] = (
        scalar[NcBy2, 1:NcBy2, north_panel_index])
    elseif panel_index == 4
        scalar_at_specific_longitude_through_panel_center[1:NcBy2] = scalar[NcBy2+1:Nc, NcBy2, west_panel_index]
        scalar_at_specific_longitude_through_panel_center[NcBy2+1:round(Int, 3Nc/2)] = scalar[1:Nc, NcBy2, panel_index]
        scalar_at_specific_longitude_through_panel_center[round(Int, 3Nc/2)+1:2*Nc] = (
        scalar[NcBy2+1, 1:NcBy2, east_panel_index])
        scalar_at_specific_longitude_through_panel_center = reverse(scalar_at_specific_longitude_through_panel_center)
    elseif panel_index == 5
        scalar_at_specific_longitude_through_panel_center[1:NcBy2] = scalar[NcBy2+1, NcBy2+1:Nc, west_panel_index]
        scalar_at_specific_longitude_through_panel_center[NcBy2+1:round(Int, 3Nc/2)] = scalar[1:Nc, NcBy2, panel_index]
        scalar_at_specific_longitude_through_panel_center[round(Int, 3Nc/2)+1:2*Nc] = (
        scalar[1:NcBy2, NcBy2, east_panel_index])
        scalar_at_specific_longitude_through_panel_center = reverse(scalar_at_specific_longitude_through_panel_center)
    end
    return scalar_at_specific_longitude_through_panel_center
end

function extract_field_at_specific_longitude_through_panel_center(grid, field, panel_index; levels = 1:1,
                                                                  read_parent_field_data = false)
    Nc = grid.Nx
    Hc = grid.Hx
    Hz = grid.Hz
    hc = read_parent_field_data ? Hc : 0
    hz = read_parent_field_data ? Hz : 0
    east_panel_index = grid.connectivity.connections[panel_index].east.from_rank
    west_panel_index = grid.connectivity.connections[panel_index].west.from_rank
    north_panel_index = grid.connectivity.connections[panel_index].north.from_rank
    south_panel_index = grid.connectivity.connections[panel_index].south.from_rank
    field_at_specific_longitude_through_panel_center = zeros(2*Nc, levels)
    NcBy2 = round(Int, Nc/2)
    if panel_index == 1
        field_at_specific_longitude_through_panel_center[1:NcBy2, levels] = (
        field[south_panel_index][NcBy2+hc, NcBy2+1+hc:Nc+hc, levels.+hz])
        field_at_specific_longitude_through_panel_center[NcBy2+1:round(Int, 3Nc/2), levels] = (
        field[panel_index][NcBy2+hc, 1+hc:Nc+hc, levels.+hz])
        field_at_specific_longitude_through_panel_center[round(Int, 3Nc/2)+1:2*Nc, levels] = (
        field[north_panel_index][1+hc:NcBy2+hc, NcBy2+1+hc, levels.+hz])
    elseif panel_index == 2
        field_at_specific_longitude_through_panel_center[1:NcBy2, levels] = (
        field[south_panel_index][NcBy2+1+hc:Nc+hc, NcBy2+1+hc, levels.+hz])
        field_at_specific_longitude_through_panel_center[NcBy2+1:round(Int, 3Nc/2), levels] = (
        field[panel_index][NcBy2+hc, 1+hc:Nc+hc, levels.+hz])
        field_at_specific_longitude_through_panel_center[round(Int, 3Nc/2)+1:2*Nc, levels] = (
        field[north_panel_index][NcBy2+hc, 1+hc:NcBy2+hc, levels.+hz])
    elseif panel_index == 4
        field_at_specific_longitude_through_panel_center[1:NcBy2, levels] = (
        field[west_panel_index][NcBy2+1+hc:Nc+hc, NcBy2+hc, levels.+hz])
        field_at_specific_longitude_through_panel_center[NcBy2+1:round(Int, 3Nc/2), levels] = (
        field[panel_index][1+hc:Nc+hc, NcBy2+hc, levels.+hz])
        field_at_specific_longitude_through_panel_center[round(Int, 3Nc/2)+1:2*Nc, levels] = (
        field[east_panel_index][NcBy2+1+hc, 1+hc:NcBy2+hc, levels.+hz])
        field_at_specific_longitude_through_panel_center = reverse(field_at_specific_longitude_through_panel_center,
                                                                   dims = 1)
    elseif panel_index == 5
        field_at_specific_longitude_through_panel_center[1:NcBy2, levels] = (
        field[west_panel_index][NcBy2+1+hc, NcBy2+1+hc:Nc+hc, levels.+hz])
        field_at_specific_longitude_through_panel_center[NcBy2+1:round(Int, 3Nc/2), levels] = (
        field[panel_index][1+hc:Nc+hc, NcBy2+hc, levels.+hz])
        field_at_specific_longitude_through_panel_center[round(Int, 3Nc/2)+1:2*Nc, levels] = (
        field[east_panel_index][1+hc:NcBy2+hc, NcBy2+hc, levels.+hz])
        field_at_specific_longitude_through_panel_center = reverse(field_at_specific_longitude_through_panel_center,
                                                                   dims = 1)
    end
    return field_at_specific_longitude_through_panel_center
end

function create_single_line_or_scatter_plot(resolution, plot_type, x, y, axis_kwargs, title, plot_kwargs, file_name;
                                            specify_x_limits = false, x_limits = [0, 0], specify_y_limits = false,
                                            y_limits = [0, 0], tight_x_axis = false, tight_y_axis = false,
                                            format = ".png")
    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1]; axis_kwargs...)

    if plot_type == "line_plot"
        lines!(ax, x, y, linewidth = plot_kwargs.linewidth, color = plot_kwargs.linecolor)
    elseif plot_type == "scatter_plot"
        scatter!(ax, x, y, marker = plot_kwargs.marker, markersize = plot_kwargs.markersize,
                 color = plot_kwargs.linecolor)
    elseif plot_type == "scatter_line_plot"
        scatterlines!(ax, x, y, linewidth = plot_kwargs.linewidth, marker = plot_kwargs.marker,
                      markersize = plot_kwargs.markersize, color = plot_kwargs.linecolor)
    end
    ax.title = title

    if specify_x_limits
        xlims!(ax, x_limits...)
    elseif tight_x_axis
        xlims!(ax, extrema(x)...)
    end

    if specify_y_limits
        ylims!(ax, y_limits...)
    elseif tight_y_axis
        ylims!(ax, extrema(y)...)
    end

    save(file_name * format, fig)
end

function test_create_single_line_or_scatter_plot()
    x = range(0, 2π, length = 100)
    y = sin.(x)

    resolution = (850, 750)

    axis_kwargs = (xlabel = "x", ylabel = "sin(x)", xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5,
                   yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1.0, titlesize = 27.5,
                   titlegap = 15, titlefont = :bold)
    title = "sin(x) vs x"
    plot_kwargs = (linewidth = 2, linecolor = :black, marker = :rect, markersize = 10)

    plot_types = ["line_plot", "scatter_plot", "scatter_line_plot"]
    file_names = ["LinePlotExample", "ScatterPlotExample", "ScatterLinePlotExample"]

    for i in 1:3
        plot_type = plot_types[i]
        file_name = file_names[i]
        create_single_line_or_scatter_plot(resolution, plot_type, x, y, axis_kwargs, title, plot_kwargs, file_name)
    end
end

function create_heat_map_or_contour_plot(resolution, plot_type, x, y, φ, axis_kwargs, title, contourlevels, cbar_kwargs,
                                         cbar_label, file_name; use_symmetric_colorrange = true,
                                         specify_plot_limits = false, plot_limits = [], format = ".png")
    fig = Figure(resolution = resolution)

    ax = Axis(fig[1,1]; axis_kwargs...)

    if specify_plot_limits
        colorrange = plot_limits
    else
        colorrange = specify_colorrange(φ; use_symmetric_colorrange = use_symmetric_colorrange)
    end
    colormap = use_symmetric_colorrange ? :balance : :amp

    if plot_type == "heat_map"
        myplot = heatmap!(ax, x, y, φ; colorrange = colorrange, colormap = colormap)
    elseif plot_type == "filled_contour_plot"
        myplot = contourf!(ax, x, y, φ; levels = range(colorrange..., length=contourlevels), colormap = colormap)
    end
    Colorbar(fig[1,2], myplot; label = cbar_label, cbar_kwargs...)
    ax.title = title
    save(file_name * format, fig)
end

function test_create_heat_map_or_contour_plot()
    resolution = (850, 750)

    nPoints = 50
    x = range(0, 2π, length = nPoints)
    y = range(0, 2π, length = nPoints)
    t = range(0, 2π, length = nPoints)

    k = 1
    l = 1

    φ = zeros(nPoints, nPoints)
    for j in 1:nPoints
        for i in 1:nPoints
            φ[i, j] = sin(k * x[i] + l * y[j])
        end
    end

    axis_kwargs = (xlabel = "x", ylabel = "y", xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5,
                   yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1,
                   titlesize = 27.5, titlegap = 15, titlefont = :bold)
    contourlevels = 50
    cbar_kwargs = (labelsize = 22.5, labelpadding = 10, ticksize = 17.5)
    cbar_label = "sin(kx + ly - ωt)"

    plot_types = ["heat_map", "filled_contour_plot"]
    titles = ["Heatmap of sin(kx + ly - ωt)", "Filled contour plot of sin(kx + ly - ωt)"]
    file_names = ["HeatMapExample", "FilledContourPlotExample"]

    for i in 1:2
        plot_type = plot_types[i]
        title = titles[i]
        file_name = file_names[i]
        create_heat_map_or_contour_plot(resolution, plot_type, x, y, φ, axis_kwargs, title, contourlevels, cbar_kwargs,
                                        cbar_label, file_name)
    end
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

function specify_colorrange(grid, φ; use_symmetric_colorrange = true, ssh = false, consider_all_levels = true,
                            levels = 1:1, read_parent_field_data = false)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    hx = read_parent_field_data ? Hx : 0
    hy = read_parent_field_data ? Hy : 0
    hz = read_parent_field_data ? Hz : 0
    
    if ssh
        consider_all_levels = false
        φ_array = zeros(Nx, Ny, 1, 6)
        levels = read_parent_field_data ? (1:1) : (Nz+1:Nz+1)
    else
        if consider_all_levels
            levels = 1:Nz .+ hz
            φ_array = zeros(Nx, Ny, Nz, 6)
        else
            levels = levels .+ hz
            φ_array = zeros(Nx, Ny, length(levels), 6)
        end
    end
    
    for region in 1:6
        φ_array[:, :, :, region] = φ[region][1+hx:Nx+hx, 1+hy:Ny+hy, levels]
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

function specify_colorrange_timeseries(grid, φ_series; Δ = 1, use_symmetric_colorrange = true, ssh = false,
                                       consider_all_levels = true, levels = 1:1, read_parent_field_data = false)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    hx = read_parent_field_data ? Hx : 0
    hy = read_parent_field_data ? Hy : 0
    hz = read_parent_field_data ? Hz : 0
    
    n = length(φ_series)
    m = floor(Int, (length(φ_series) - 1)/Δ + 1)
    
    if ssh
        consider_all_levels = false
        φ_series_array = zeros(Nx, Ny, 1, 6, m)
        levels = read_parent_field_data ? (1:1) : (Nz+1:Nz+1)
    else
        if consider_all_levels
            levels = 1:Nz .+ hz
            φ_series_array = zeros(Nx, Ny, Nz, 6, m)
        else
            levels = levels .+ hz
            φ_series_array = zeros(Nx, Ny, length(levels), 6, m)
        end
    end
    
    j = 0
    for i in 1:Δ:n
        j += 1
        for region in 1:6
            φ_series_array[:, :, :, region, j] = φ_series[i][region][1+hx:Nx+hx, 1+hy:Ny+hy, levels]
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
    colormap = use_symmetric_colorrange ? :balance : :amp

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

function panel_wise_visualization_with_halos(grid, field; k = 1, use_symmetric_colorrange = true, ssh = false,
                                             consider_all_levels = true, levels = k:k, read_parent_field_data = false,
                                             specify_plot_limits = false, plot_limits = [])
    fig = Figure(resolution = (2450, 1400))

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, aspect = 1.0, 
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 27.5, titlegap = 15, titlefont = :bold,
                   xlabel = "Local x direction", ylabel = "Local y direction")
    
    if specify_plot_limits
        colorrange = plot_limits
    else
        colorrange = specify_colorrange(grid, field; use_symmetric_colorrange = use_symmetric_colorrange, ssh = ssh,
                                        consider_all_levels = consider_all_levels, levels = levels,
                                        read_parent_field_data = read_parent_field_data)
    end
    colormap = use_symmetric_colorrange ? :balance : :amp
    
    hz = read_parent_field_data ? grid.Hz : 0

    if ssh
        k = read_parent_field_data ? 1 : grid.Nz+1
    else
        k += hz
    end

    ax_1 = Axis(fig[3, 1]; title = "Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, parent(getregion(field, 1)[:, :, k]); colorrange, colormap)
    Colorbar(fig[3, 2], hm_1)

    ax_2 = Axis(fig[3, 3]; title = "Panel 2", axis_kwargs...)
    hm_2 = heatmap!(ax_2, parent(getregion(field, 2)[:, :, k]); colorrange, colormap)
    Colorbar(fig[3, 4], hm_2)

    ax_3 = Axis(fig[2, 3]; title = "Panel 3", axis_kwargs...)
    hm_3 = heatmap!(ax_3, parent(getregion(field, 3)[:, :, k]); colorrange, colormap)
    Colorbar(fig[2, 4], hm_3)

    ax_4 = Axis(fig[2, 5]; title = "Panel 4", axis_kwargs...)
    hm_4 = heatmap!(ax_4, parent(getregion(field, 4)[:, :, k]); colorrange, colormap)
    Colorbar(fig[2, 6], hm_4)

    ax_5 = Axis(fig[1, 5]; title = "Panel 5", axis_kwargs...)
    hm_5 = heatmap!(ax_5, parent(getregion(field, 5)[:, :, k]); colorrange, colormap)
    Colorbar(fig[1, 6], hm_5)

    ax_6 = Axis(fig[1, 7]; title = "Panel 6", axis_kwargs...)
    hm_6 = heatmap!(ax_6, parent(getregion(field, 6)[:, :, k]); colorrange, colormap)
    Colorbar(fig[1, 8], hm_6)

    return fig
end

function panel_wise_visualization(grid, field; k = 1, use_symmetric_colorrange = true, ssh = false,
                                  consider_all_levels = true, levels = k:k, read_parent_field_data = false,
                                  specify_plot_limits = false, plot_limits = [])
    fig = Figure(resolution = (2450, 1400))
    
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    hx = read_parent_field_data ? Hx : 0
    hy = read_parent_field_data ? Hy : 0
    hz = read_parent_field_data ? Hz : 0

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, aspect = 1.0, 
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 27.5, titlegap = 15, titlefont = :bold,
                   xlabel = "Local x direction", ylabel = "Local y direction")
    
    if specify_plot_limits
        colorrange = plot_limits
    else
        colorrange = specify_colorrange(grid, field; use_symmetric_colorrange = use_symmetric_colorrange, ssh = ssh,
                                        consider_all_levels = consider_all_levels, levels = levels)
    end
    colormap = use_symmetric_colorrange ? :balance : :amp
    
    if ssh
        k = read_parent_field_data ? 1 : Nz+1
    else
        k += hz
    end

    ax_1 = Axis(fig[3, 1]; title = "Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, parent(getregion(field, 1)[1+hx:Nx+hx, 1+hy:Ny+hy, k]); colorrange, colormap)
    Colorbar(fig[3, 2], hm_1)

    ax_2 = Axis(fig[3, 3]; title = "Panel 2", axis_kwargs...)
    hm_2 = heatmap!(ax_2, parent(getregion(field, 2)[1+hx:Nx+hx, 1+hy:Ny+hy, k]); colorrange, colormap)
    Colorbar(fig[3, 4], hm_2)

    ax_3 = Axis(fig[2, 3]; title = "Panel 3", axis_kwargs...)
    hm_3 = heatmap!(ax_3, parent(getregion(field, 3)[1+hx:Nx+hx, 1+hy:Ny+hy, k]); colorrange, colormap)
    Colorbar(fig[2, 4], hm_3)

    ax_4 = Axis(fig[2, 5]; title = "Panel 4", axis_kwargs...)
    hm_4 = heatmap!(ax_4, parent(getregion(field, 4)[1+hx:Nx+hx, 1+hy:Ny+hy, k]); colorrange, colormap)
    Colorbar(fig[2, 6], hm_4)

    ax_5 = Axis(fig[1, 5]; title = "Panel 5", axis_kwargs...)
    hm_5 = heatmap!(ax_5, parent(getregion(field, 5)[1+hx:Nx+hx, 1+hy:Ny+hy, k]); colorrange, colormap)
    Colorbar(fig[1, 6], hm_5)

    ax_6 = Axis(fig[1, 7]; title = "Panel 6", axis_kwargs...)
    hm_6 = heatmap!(ax_6, parent(getregion(field, 6)[1+hx:Nx+hx, 1+hy:Ny+hy, k]); colorrange, colormap)
    Colorbar(fig[1, 8], hm_6)

    return fig
end

function geo_heatlatlon_visualization(grid, field, title; k = 1, use_symmetric_colorrange = true, ssh = false,
                                      consider_all_levels = true, levels = k:k, cbar_label = "",
                                      specify_plot_limits = false, plot_limits = [])
# Ensure that field is a CubedSphereField interpolated to cell centers.
    fig = Figure(resolution = (1350, 650))

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5,
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 25, titlegap = 15, titlefont = :bold)

    if ssh
        consider_all_levels = false
        k = 1
        levels = k:k
    end

    if specify_plot_limits
        colorrange = plot_limits
    else
        colorrange = specify_colorrange(grid, field; use_symmetric_colorrange = use_symmetric_colorrange, ssh = ssh,
                                        consider_all_levels = consider_all_levels, levels = levels)
    end
    colormap = use_symmetric_colorrange ? :balance : :amp

    ax = GeoAxis(fig[1, 1]; coastlines = true, lonlims = automatic, title = title, axis_kwargs...)
    heatlatlon!(ax, field, k; colorrange, colormap)

    Colorbar(fig[1, 2], limits = colorrange, colormap = colormap, label = cbar_label, labelsize = 22.5,
             labelpadding = 10, ticksize = 17.5, width = 25, height = Relative(0.9))

    colsize!(fig.layout, 1, Auto(0.8))
    colgap!(fig.layout, 75)

    return fig
end

function create_single_line_or_scatter_plot_animation(resolution, plot_type, x, y_series, axis_kwargs, title_prefix,
                                                      plot_kwargs, framerate, filename; start_index = 1,
                                                      specify_x_limits = false, x_limits = [0, 0],
                                                      specify_y_limits = false, y_limits = [0, 0],
                                                      use_symmetric_range = true, use_prettytimes = false,
                                                      prettytimes = [], tight_x_axis = false, format = ".mp4")
    n = Observable(start_index)
    y = @lift y_series[$n, :]
    use_prettytimes ? (prettytime = @lift prettytimes[$n]) : nothing

    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1]; axis_kwargs...)
    ax.title = use_prettytimes ? (title_prefix * " after " * prettytime[]) : title_prefix

    if !specify_y_limits
        y_limits = specify_colorrange(y_series; use_symmetric_colorrange = use_symmetric_range)
    end

    if plot_type == "line_plot"
        lines!(ax, x, y, linewidth = plot_kwargs.linewidth, color = plot_kwargs.linecolor)
    elseif plot_type == "scatter_plot"
        scatter!(ax, x, y, marker = plot_kwargs.marker, markersize = plot_kwargs.markersize,
                 color = plot_kwargs.linecolor)
    elseif plot_type == "scatter_line_plot"
        scatterlines!(ax, x, y, linewidth = plot_kwargs.linewidth, marker = plot_kwargs.marker,
                      markersize = plot_kwargs.markersize, color = plot_kwargs.linecolor)
    end

    if specify_x_limits
        xlims!(ax, x_limits...)
    elseif tight_x_axis
        xlims!(ax, extrema(x)...)
    end

    ylims!(ax, y_limits...)

    frames = 1:size(y_series, 1)
    CairoMakie.record(fig, filename * format, frames, framerate = framerate) do i
        msg = string("Plotting frame ", i, " of ", frames[end])
        print(msg * " \r")
        n[] = i
        ax.title = use_prettytimes ? (title_prefix * " after " * prettytime[]) : title_prefix
    end
end

function test_create_single_line_or_scatter_plot_animation()
    x = range(0, 2π, length = 100)
    t = range(0, 2π, length = 100)

    prettytimes = [@sprintf("%.3f seconds", t[i]) for i in 1:length(t)]
    y_series = [sin(x[i] - t[k]) for k in 1:100, i in 1:100]

    resolution = (850, 750)
    axis_kwargs = (xlabel = "x", ylabel = "sin(x)", xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5,
                   yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1.0, titlesize = 27.5,
                   titlegap = 15, titlefont = :bold)
    title = "sin(x) vs x"
    plot_kwargs = (linewidth = 2, linecolor = :black, marker = :rect, markersize = 10)
    framerate = 10

    plot_types = ["line_plot", "scatter_plot", "scatter_line_plot"]
    file_names = ["LinePlotAnimationExample", "ScatterPlotAnimationExample", "ScatterLinePlotAnimationExample"]

    for i in 1:3
        plot_type = plot_types[i]
        file_name = file_names[i]
        create_single_line_or_scatter_plot_animation(resolution, plot_type, x, y_series, axis_kwargs, title,
                                                     plot_kwargs, framerate, file_name; use_prettytimes = true,
                                                     prettytimes = prettytimes)
    end
end

function create_heat_map_or_contour_plot_animation(resolution, plot_type, x, y, φ_series, axis_kwargs, title_prefix,
                                                   contourlevels, cbar_kwargs, cbar_label, framerate, filename;
                                                   start_index = 1, use_symmetric_colorrange = true,
                                                   specify_plot_limits = false, plot_limits = [],
                                                   use_prettytimes = false, prettytimes = [], format = ".mp4")
    n = Observable(start_index)
    φ = @lift φ_series[$n, :, :]
    use_prettytimes ? (prettytime = @lift prettytimes[$n]) : nothing

    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1]; axis_kwargs...)
    ax.title = use_prettytimes ? (title_prefix * " after " * prettytime[]) : title_prefix

    if specify_plot_limits
        colorrange = plot_limits
    else
        colorrange = specify_colorrange(φ_series; use_symmetric_colorrange = use_symmetric_colorrange)
    end
    colormap = use_symmetric_colorrange ? :balance : :amp

    if plot_type == "heat_map"
        myplot = heatmap!(ax, x, y, φ; colorrange = colorrange, colormap = colormap)
    elseif plot_type == "filled_contour_plot"
        myplot = contourf!(ax, x, y, φ; levels = range(colorrange..., length=contourlevels), colormap = colormap)
    end
    Colorbar(fig[1,2], myplot; label = cbar_label, cbar_kwargs...)

    frames = 1:size(φ_series, 1)
    CairoMakie.record(fig, filename * format, frames, framerate = framerate) do i
        msg = string("Plotting frame ", i, " of ", frames[end])
        print(msg * " \r")
        n[] = i
        ax.title = use_prettytimes ? (title_prefix * " after " * prettytime[]) : title_prefix
    end
end

function test_create_heat_map_or_contour_plot_animation()
    resolution = (850, 750)

    nPoints = 50
    x = range(0, 2π, length = nPoints)
    y = range(0, 2π, length = nPoints)
    t = range(0, 2π, length = nPoints)

    k = 1
    l = 1
    ω = 1

    prettytimes = [@sprintf("%.3f seconds", t[i]) for i in 1:length(t)]
    φ_series = zeros(nPoints, nPoints, nPoints)
    for j in 1:nPoints
        for i in 1:nPoints
            for m in 1:nPoints
                φ_series[m, i, j] = sin(k * x[i] + l * y[j] - ω * t[m])
            end
        end
    end

    axis_kwargs = (xlabel = "x", ylabel = "y", xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5,
                   yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1,
                   title = "sin(kx + ly - ωt)", titlesize = 27.5, titlegap = 15, titlefont = :bold)
    contourlevels = 50
    cbar_kwargs = (labelsize = 22.5, labelpadding = 10, ticksize = 17.5)
    cbar_label = "sin(kx + ly - ωt)"
    framerate = 10

    plot_types = ["heat_map", "filled_contour_plot"]
    titles = ["Heatmap of sin(kx + ly - ωt)", "Filled contour plot of sin(kx + ly - ωt)"]
    file_names = ["sine_wave_heat_map_animation", "sine_wave_filled_contour_plot_animation"]

    for i in 1:2
        plot_type = plot_types[i]
        title = titles[i]
        file_name = file_names[i]
        create_heat_map_or_contour_plot_animation(resolution, plot_type, x, y, φ_series, axis_kwargs, title, contourlevels,
                                                  cbar_kwargs, cbar_label, framerate, file_name; use_prettytimes = true,
                                                  prettytimes = prettytimes)
    end
end

function create_panel_wise_visualization_animation_with_halos(grid, φ_series, framerate, filename; start_index = 1,
                                                              k = 1, use_symmetric_colorrange = true, ssh = false,
                                                              consider_all_levels = true, levels = k:k,
                                                              specify_plot_limits = false, plot_limits = [],
                                                              format = ".mp4")
    n = Observable(start_index) # the current index
    φ = @lift φ_series[$n]

    # Create the initial visualization.
    fig = Figure(resolution = (2450, 1400))
    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, aspect = 1.0,
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 27.5, titlegap = 15, titlefont = :bold,
                   xlabel = "Local x direction", ylabel = "Local y direction")

    if ssh
        consider_all_levels = false
        k = grid.Nz+1
        levels = k:k
    end

    if specify_plot_limits
        colorrange = plot_limits
    else
        colorrange = specify_colorrange_timeseries(grid, φ_series; use_symmetric_colorrange = use_symmetric_colorrange,
                                                   ssh = ssh, consider_all_levels = consider_all_levels,
                                                   levels = levels)
    end
    colormap = use_symmetric_colorrange ? :balance : :amp

    ax_1 = Axis(fig[3, 1]; title = "Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, parent(φ[][1][:, :, k]); colorrange, colormap)
    Colorbar(fig[3, 2], hm_1)

    ax_2 = Axis(fig[3, 3]; title = "Panel 2", axis_kwargs...)
    hm_2 = heatmap!(ax_2, parent(φ[][2][:, :, k]); colorrange, colormap)
    Colorbar(fig[3, 4], hm_2)

    ax_3 = Axis(fig[2, 3]; title = "Panel 3", axis_kwargs...)
    hm_3 = heatmap!(ax_3, parent(φ[][3][:, :, k]); colorrange, colormap)
    Colorbar(fig[2, 4], hm_3)

    ax_4 = Axis(fig[2, 5]; title = "Panel 4", axis_kwargs...)
    hm_4 = heatmap!(ax_4, parent(φ[][4][:, :, k]); colorrange, colormap)
    Colorbar(fig[2, 6], hm_4)

    ax_5 = Axis(fig[1, 5]; title = "Panel 5", axis_kwargs...)
    hm_5 = heatmap!(ax_5, parent(φ[][5][:, :, k]); colorrange, colormap)
    Colorbar(fig[1, 6], hm_5)

    ax_6 = Axis(fig[1, 7]; title = "Panel 6", axis_kwargs...)
    hm_6 = heatmap!(ax_6, parent(φ[][6][:, :, k]); colorrange, colormap)
    Colorbar(fig[1, 8], hm_6)

    frames = 1:length(φ_series)
    CairoMakie.record(fig, filename * format, frames, framerate = framerate) do i
        msg = string("Plotting frame ", i, " of ", frames[end])
        print(msg * " \r")

        φ[] = φ_series[i]

        hm_1 = heatmap!(ax_1, parent(φ[][1][:, :, k]); colorrange, colormap)
        Colorbar(fig[3, 2], hm_1)

        hm_2 = heatmap!(ax_2, parent(φ[][2][:, :, k]); colorrange, colormap)
        Colorbar(fig[3, 4], hm_2)

        hm_3 = heatmap!(ax_3, parent(φ[][3][:, :, k]); colorrange, colormap)
        Colorbar(fig[2, 4], hm_3)

        hm_4 = heatmap!(ax_4, parent(φ[][4][:, :, k]); colorrange, colormap)
        Colorbar(fig[2, 6], hm_4)

        hm_5 = heatmap!(ax_5, parent(φ[][5][:, :, k]); colorrange, colormap)
        Colorbar(fig[1, 6], hm_5)

        hm_6 = heatmap!(ax_6, parent(φ[][6][:, :, k]); colorrange, colormap)
        Colorbar(fig[1, 8], hm_6)
    end
end

function create_panel_wise_visualization_animation(grid, φ_series, framerate, filename; start_index = 1, k = 1,
                                                   use_symmetric_colorrange = true, ssh = false,
                                                   consider_all_levels = true, levels = k:k,
                                                   read_parent_field_data = false, specify_plot_limits = false,
                                                   plot_limits = [], format = ".mp4")
    Nx, Ny, Nz = size(grid)

    n = Observable(start_index) # the current index
    φ = @lift φ_series[$n]

    # Create the initial visualization.
    fig = Figure(resolution = (2450, 1400))
    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, aspect = 1.0,
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 27.5, titlegap = 15, titlefont = :bold,
                   xlabel = "Local x direction", ylabel = "Local y direction")

    if ssh
        consider_all_levels = false
        k = Nz+1
        levels = k:k
    end

    if specify_plot_limits
        colorrange = plot_limits
    else
        colorrange = specify_colorrange_timeseries(grid, φ_series; use_symmetric_colorrange = use_symmetric_colorrange,
                                                   ssh = ssh, consider_all_levels = consider_all_levels,
                                                   levels = levels)
    end
    colormap = use_symmetric_colorrange ? :balance : :amp

    ax_1 = Axis(fig[3, 1]; title="Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, φ[][1][1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[3, 2], hm_1)

    ax_2 = Axis(fig[3, 3]; title="Panel 2", axis_kwargs...)
    hm_2 = heatmap!(ax_2, φ[][2][1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[3, 4], hm_2)

    ax_3 = Axis(fig[2, 3]; title="Panel 3", axis_kwargs...)
    hm_3 = heatmap!(ax_3, φ[][3][1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[2, 4], hm_3)

    ax_4 = Axis(fig[2, 5]; title="Panel 4", axis_kwargs...)
    hm_4 = heatmap!(ax_4, φ[][4][1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[2, 6], hm_4)

    ax_5 = Axis(fig[1, 5]; title="Panel 5", axis_kwargs...)
    hm_5 = heatmap!(ax_5, φ[][5][1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[1, 6], hm_5)

    ax_6 = Axis(fig[1, 7]; title="Panel 6", axis_kwargs...)
    hm_6 = heatmap!(ax_6, φ[][6][1:Nx, 1:Ny, k]; colorrange, colormap)
    Colorbar(fig[1, 8], hm_6)

    frames = 1:length(φ_series)
    CairoMakie.record(fig, filename * format, frames, framerate = framerate) do i
        msg = string("Plotting frame ", i, " of ", frames[end])
        print(msg * " \r")

        φ[] = φ_series[i]

        hm_1 = heatmap!(ax_1, φ[][1][1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[3, 2], hm_1)

        hm_2 = heatmap!(ax_2, φ[][2][1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[3, 4], hm_2)

        hm_3 = heatmap!(ax_3, φ[][3][1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[2, 4], hm_3)

        hm_4 = heatmap!(ax_4, φ[][4][1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[2, 6], hm_4)

        hm_5 = heatmap!(ax_5, φ[][5][1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[1, 6], hm_5)

        hm_6 = heatmap!(ax_6, φ[][6][1:Nx, 1:Ny, k]; colorrange, colormap)
        Colorbar(fig[1, 8], hm_6)
    end
end

function geo_heatlatlon_visualization_animation(grid, fields, location, prettytimes, title_prefix, filename;
                                                start_index = 1, k = 1, use_symmetric_colorrange = true, ssh = false,
                                                consider_all_levels = true, levels = k:k, cbar_label = "",
                                                specify_plot_limits = false, plot_limits = [], framerate = 10,
                                                format = ".mp4")
    n = Observable(start_index)
    field = @lift fields[$n]
    # Ensure that field is a CubedSphereField interpolated to cell centers.
    prettytime = @lift prettytimes[$n]

    fig = Figure(resolution=(1350, 650))

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5,
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 25, titlegap = 15, titlefont = :bold)

    if ssh
        consider_all_levels = false
        k = 1
        levels = k:k
    end

    if specify_plot_limits
        colorrange = plot_limits
    else
        colorrange = specify_colorrange_timeseries(grid, fields; use_symmetric_colorrange = use_symmetric_colorrange,
                                                   ssh = ssh, consider_all_levels = consider_all_levels,
                                                   levels = levels)
    end
    colormap = use_symmetric_colorrange ? :balance : :amp

    ax = GeoAxis(fig[1, 1]; coastlines = true, lonlims = automatic, axis_kwargs...)
    ax.title = title_prefix * " after " * prettytime[]

    interpolated_field = interpolate_cubed_sphere_field_to_cell_centers(grid, field[], location; levels = k:k)
    heatlatlon!(ax, interpolated_field, k; colorrange, colormap)

    Colorbar(fig[1, 2], limits = colorrange, colormap = colormap, label = cbar_label, labelsize = 22.5,
             labelpadding = 10, ticksize = 17.5, width = 25, height = Relative(0.9))
    colsize!(fig.layout, 1, Auto(0.8))
    colgap!(fig.layout, 75)

    frames = 1:length(fields)
    CairoMakie.record(fig, filename * format, frames, framerate = framerate) do i
        msg = string("Plotting frame ", i, " of ", frames[end])
        print(msg * " \r")

        field[] = fields[i]
        prettytime[] = prettytimes[i]

        # Update the title of the plot
        ax.title = title_prefix * " after " * prettytime[]

        # Update the plot
        interpolated_field = interpolate_cubed_sphere_field_to_cell_centers(grid, field[], location; levels = k:k)
        heatlatlon!(ax, interpolated_field, k; colorrange, colormap)

        Colorbar(fig[1, 2], limits = colorrange, colormap = colormap, label = cbar_label, labelsize = 22.5,
                 labelpadding = 10, ticksize = 17.5, width = 25, height = Relative(0.9))
        colsize!(fig.layout, 1, Auto(0.8))
        colgap!(fig.layout, 75)
    end
end
