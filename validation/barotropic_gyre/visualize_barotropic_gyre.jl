# # Barotropic gyre

using Oceananigans.Grids
using Oceananigans.Utils: prettytime, hours, day, days, years

using Statistics
using JLD2
using Printf
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

function visualize_barotropic_gyre(filepath)

    file = jldopen(filepath)

    Nx = file["grid/Nx"]
    Ny = file["grid/Ny"]

    # A spherical domain
    grid = LatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                 longitude = (-30, 30),
                                 latitude = (15, 75),
                                 z = (-4000, 0))

    iterations = parse.(Int, keys(file["timeseries/t"]))

    xu, yu, zu = geographic2cartesian(xnodes(Face,   grid), ynodes(Center, grid))
    xv, yv, zv = geographic2cartesian(xnodes(Center, grid), ynodes(Face,   grid))
    xc, yc, zc = geographic2cartesian(xnodes(Center, grid), ynodes(Center, grid))

    iter = Observable(0)

    plot_title = @lift @sprintf("Barotropic gyre: time = %s", prettytime(file["timeseries/t/" * string($iter)]))

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
        wireframe!(ax, Sphere(Point3f(0), 0.99f0), show_axis=false)
        surface!(ax, x[n], y[n], z[n], color=var, colormap=:balance) #, colorrange=clims[n])
        rotate_cam!(ax.scene, (0, 3π/4, 0))
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

#Nx = 720
#Ny = 720

#Nx = 360
#Ny = 360

Nx = 60
Ny = 60

output_prefix = "barotropic_gyre_Nx$(Nx)_Ny$(Ny)"
filepath = output_prefix * ".jld2"

visualize_barotropic_gyre(filepath)
