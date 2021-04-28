using Oceananigans.Grids
using Oceananigans.Utils: prettytime, hours, day, days, years

using Statistics
using JLD2
using Printf

#=
using Makie
using GLMakie
function geographic2cartesian(λ, φ, r=1)
    Nλ = length(λ)
    Nφ = length(φ)

    λ = repeat(reshape(λ, Nλ, 1), 1, Nφ) 
    φ = repeat(reshape(φ, 1, Nφ), Nλ, 1)

    λ_azimuthal = λ .+ 180  # Convert to λ ∈ [0°, 360°]
    φ_azimuthal = 90 .- φ   # Convert to φ ∈ [0°, 180°] (0° at north pole)

    x = @. r * cosd(λ_azimuthal) * sind(φ_azimuthal)
    y = @. r * sind(λ_azimuthal) * sind(φ_azimuthal)
    z = @. r * cosd(φ_azimuthal)

    return x, y, z
end

function visualize_makie(filepath)

    file = jldopen(filepath)

    Nx = file["grid/Nx"]
    Ny = file["grid/Ny"]

    # A spherical domain
    grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                        longitude = (-30, 30),
                                        latitude = (15, 75),
                                        z = (-4000, 0))

    iterations = parse.(Int, keys(file["timeseries/t"]))

    xu, yu, zu = geographic2cartesian(xnodes(Face,   grid), ynodes(Center, grid))
    xv, yv, zv = geographic2cartesian(xnodes(Center, grid), ynodes(Face,   grid))
    xc, yc, zc = geographic2cartesian(xnodes(Center, grid), ynodes(Center, grid))

    iter = Node(0)

    plot_title = @lift @sprintf("Barotropic gyre: time = %s", prettytime(file["timeseries/t/" * string($iter)]))

    u = @lift file["timeseries/u/" * string($iter)][:, :, 1]
    v = @lift file["timeseries/v/" * string($iter)][:, :, 1]
    η = @lift file["timeseries/η/" * string($iter)][:, :, 1]

    u = @lift file["timeseries/u/" * string($iter)][:, :, 1]
    v = @lift file["timeseries/v/" * string($iter)][:, :, 1]
    η = @lift file["timeseries/η/" * string($iter)][:, :, 1]

    fig = Figure(resolution = (2160, 1080))

    x = (xu, xv, xc)
    y = (yu, yv, yc)
    z = (zu, zv, zc)

    statenames = ["u", "v", "η"]
    for (n, var) in enumerate([u, v, η])
        ax = fig[3:7, 3n-2:3n] = LScene(fig) # make plot area wider
        wireframe!(ax, Sphere(Point3f0(0), 0.99f0), show_axis=false)
        surface!(ax, x[n], y[n], z[n], color=var, colormap=:balance) #, colorrange=clims[n])
        rotate_cam!(ax.scene, (3π/4, -π/8, 0))
        zoom!(ax.scene, (0, 0, 0), 5, false)
        fig[2, 2 + 3*(n-1)] = Label(fig, statenames[n], textsize = 50) # put names in center
    end

    supertitle = fig[0, :] = Label(fig, plot_title, textsize=50)

    record(fig, output_prefix * ".mp4", iterations, framerate=30) do i
        @info "Animating iteration $i/$(iterations[end])..."
        iter[] = i
    end

    close(file)

    return nothing
end
=#

using Plots

function visualize_plots(filepath)

    file = jldopen(filepath)

    Nx = file["grid/Nx"]
    Ny = file["grid/Ny"]
    Lλ = file["grid/Lx"]
    Lφ = file["grid/Ly"]
    Lz = file["grid/Lz"]

    # A spherical domain
    grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                        longitude = (-180, 180),
                                        latitude = (-Lφ/2, Lφ/2),
                                        z = (-Lz, 0))

    iterations = parse.(Int, keys(file["timeseries/t"]))

    anim = @animate for (i, iter) in enumerate(iterations)
        u = file["timeseries/u/$iter"][:, :, 1]
        Plots.heatmap(u')
    end

    close(file)

    mp4(anim, "eddying_strip.mp4", fps = 8) # hide

    return nothing
end

#Nx = 720
#Ny = 720

Nx = 3600
Ny = 3600

output_prefix = "barotropic_gyre_Nx$(Nx)_Ny$(Ny)"
filepath = output_prefix * ".jld2"

visualize_plots(filepath)
