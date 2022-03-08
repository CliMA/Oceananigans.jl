using Oceananigans
using Oceananigans.Utils: prettytime
using JLD2
using GLMakie

function variable_range(file, variable)
    max_var = -Inf
    min_var = Inf
    for i in collect(parse.(Int, keys(file["timeseries/t"])))
        max_var = max(max_var, maximum(file["timeseries/$(variable)/" * string(i)][:, :, 1]))
        min_var = min(max_var, minimum(file["timeseries/$(variable)/" * string(i)][:, :, 1]))
    end
    return (min_var * 0.5, max_var * 0.5)
end

function visualize_spherical_field(filepath, variable, range = nothing)

    @show output_prefix = basename(filepath)[1:end-5]

    file = jldopen(filepath)

    iterations = parse.(Int, keys(file["timeseries/t"]))
    
    if range isa Nothing
        range = variable_range(file, variable)
    end
    Nx = file["grid/Nx"]
    Ny = file["grid/Ny"]

    grid = LatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                 radius = 1,
                                 latitude = (-80, 80),
                                 longitude = (-180, 180),
                                 z = (-1, 0))

    λ = xnodes(Face, grid)
    φ = ynodes(Center, grid)

    λ = repeat(reshape(λ, Nx, 1), 1, Ny)
    φ = repeat(reshape(φ, 1, Ny), Nx, 1)

    λ_azimuthal = λ .+ 180  # Convert to λ ∈ [0°, 360°]
    φ_azimuthal = 90 .- φ   # Convert to φ ∈ [0°, 180°] (0° at north pole)

    iter = Observable(0)

    plot_title = @lift "vorticity at: time = $(prettytime(file["timeseries/t/" * string($iter)]))"

    var = @lift file["timeseries/$(variable)/" * string($iter)][:, :, 1]
    
    # Plot on the unit sphere to align with the spherical wireframe.
    x = @. cosd(λ_azimuthal) * sind(φ_azimuthal)
    y = @. sind(λ_azimuthal) * sind(φ_azimuthal)
    z = @. cosd(φ_azimuthal)

    fig = Figure(resolution = (2000, 2000))
    
    fontsize_theme = Theme(fontsize = 25)
    set_theme!(fontsize_theme)
    
    ax = fig[1, 1] = LScene(fig)
    wireframe!(ax, Sphere(Point3f0(0), 0.99f0), show_axis=false)
    hm = surface!(ax, x, y, z, color=var, colormap=:thermal, colorrange=range)
    rotate_cam!(ax.scene, (π/5, π/6, 0))
    
    cb = Colorbar(fig[1, 2], hm, label=plot_title)
    cb.height = Relative(2/3)
    cb.width = 20


    record(fig, output_prefix * ".mp4", iterations, framerate=80) do i
        @info "Animating iteration $i/$(iterations[end])..."
        iter[] = i
    end

    return nothing
end


function visualize_cartesian_field(filepath, variable, range = nothing)

    @show output_prefix = basename(filepath)[1:end-5]

    file = jldopen(filepath)

    iterations = parse.(Int, keys(file["timeseries/t"]))
    
    if range isa Nothing
        range = variable_range(file, variable)
    end
    Nx = file["grid/Nx"]
    Ny = file["grid/Ny"]

    grid = LatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                 radius = 1,
                                 latitude = (-80, 80),
                                 longitude = (-180, 180),
                                 z = (-1, 0))

    λ = xnodes(Face, grid)
    φ = ynodes(Center, grid)

    iter = Observable(0)

    plot_title = @lift "vorticity at: time = $(prettytime(file["timeseries/t/" * string($iter)]))"

    var = @lift file["timeseries/$(variable)/" * string($iter)][:, :, 1]
    
    fig = Figure(resolution = (2000, 2000))
    
    fontsize_theme = Theme(fontsize = 25)
    set_theme!(fontsize_theme)
    
    ax = fig[1, 1] = LScene(fig)
    hm = heatmap!(ax, λ, φ, var, colormap=:thermal, colorrange=range)
    
    cb = Colorbar(fig[1, 2], hm, label=plot_title)
    cb.height = Relative(2/3)
    cb.width = 20


    record(fig, output_prefix * ".mp4", iterations, framerate=80) do i
        @info "Animating iteration $i/$(iterations[end])..."
        iter[] = i
    end

    return nothing
end


function visualize_cartesian_comparison(filepath1, filepath2, title1, title2, variable, crange = nothing)

    @show output_prefix = "comparison"

    file1 = jldopen(filepath1)
    file2 = jldopen(filepath2)

    iterations = parse.(Int, keys(file1["timeseries/t"]))
    
    iterations = iterations[1:4:end]
    if crange isa Nothing
        crange = variable_range(file1, variable)
    end
    Nx = file1["grid/Nx"]
    Ny = file1["grid/Ny"]

    grid = LatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                 radius = 1,
                                 latitude = (-80, 80),
                                 longitude = (-180, 180),
                                 z = (-1, 0))

    λ = xnodes(Face, grid)
    φ = ynodes(Center, grid)

    iter = Observable(0)

    plot_title = @lift "vorticity at: time = $(prettytime(file1["timeseries/t/" * string($iter)]))"

    var1 = @lift file1["timeseries/$(variable)/" * string($iter)][:, :, 1]
    var2 = @lift file2["timeseries/$(variable)/" * string($iter)][:, :, 1]
    
    fig = Figure(resolution = (3500, 1000))
    
    fontsize_theme = Theme(fontsize = 25)
    set_theme!(fontsize_theme)
    
    ax = Axis(fig[1, 1], title = title1)
    hm = heatmap!(ax, var1, colormap=:thermal, colorrange=crange)
    ct = contour!(ax, var1, levels = range(crange[1], crange[2], length = 10), color = :black, interpolate = true)
    
    cb = Colorbar(fig[1, 2], hm, label=plot_title)
    cb.height = Relative(2/3)
    cb.width = 20

    ax = Axis(fig[1, 3], title = title2)
    hm = heatmap!(ax, var2, colormap=:thermal, colorrange=crange)
    ct = contour!(ax, var2, levels = range(crange[1], crange[2], length = 10), color = :black, interpolate = true)

    cb = Colorbar(fig[1, 4], hm, label=plot_title)
    cb.height = Relative(2/3)
    cb.width = 20

    record(fig, output_prefix * ".mp4", iterations, framerate=150) do i
        @info "Animating iteration $i/$(iterations[end])..."
        iter[] = i
    end

    return nothing
end