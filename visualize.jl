using Oceananigans.Grids
using Oceananigans.Utils: prettytime, hours, day, days, years

using Statistics
using JLD2
using Printf
using GLMakie
#using CairoMakie

function geographic2cartesian(λ, φ; r=1)
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

function visualize_makie(output_prefix)

    filepath = output_prefix * ".jld2"

    file = jldopen(filepath)

    Nx = file["grid/underlying_grid/Nx"]
    Ny = file["grid/underlying_grid/Ny"]
    Lλ = file["grid/underlying_grid/Lx"]
    Lφ = file["grid/underlying_grid/Ly"]
    Lz = file["grid/underlying_grid/Lz"]

    # A spherical domain
    grid = LatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                 longitude = (-180, 180),
                                 latitude = (-Lφ/2, Lφ/2),
                                 z = (-Lz, 0))

    iterations = parse.(Int, keys(file["timeseries/t"]))

    λ, ϕ, r = nodes((Face, Face, Center), grid)
    x, y, z = geographic2cartesian(λ, ϕ, r=1.01)

    iter = Observable(0)
    #ζ′ = @lift file["timeseries/ζ/" * string($iter)][:, :, 1]
    ζ′ = file["timeseries/ζ/" * string($iter)][:, :, 1]
    
    fig = Figure(resolution = (2000, 2000))

    clims = @lift begin
        scale = 1.5e-1
        min_clim = 2e-5
        max_ζ = maximum(abs, file["timeseries/ζ/" * string($iter)][:, :, 1])
        scale * max_ζ < min_clim && (-min_clim, min_clim)
        scale .* (-max_ζ, max_ζ)
    end

    ax = fig[:, :] = LScene(fig) # make plot area wider
    #wireframe!(ax, Sphere(Point3f0(0), 1f0), show_axis=false)
    wireframe!(ax, Sphere(Point3f(0), 1))
    surface!(ax, x, y, z, color=ζ′, colormap=:blues, colorrange=clims)
    rotate_cam!(ax.scene, (π/4, π/6, 0))
    #zoom!(ax.scene, (0, 0, 0), 5, false)
    zoom!(ax.scene, cameracontrols(ax.scene), 5)
    
    plot_title = @lift @sprintf("Vertical vorticity in decaying, rotating, barotropic turbulence at time = %s",
                                prettytime(file["timeseries/t/" * string($iter)]))
    supertitle = fig[0, :] = Label(fig, plot_title, textsize=50)

    display(fig)

    record(fig, output_prefix * ".mp4", iterations, framerate=12) do i
        @info "Plotting iteration $i of $(iterations[end])..."
        iter[] = i
    end

    return nothing
end

output_prefix = "near_global_lat_lon_1440_600__fine_surface"

visualize_makie(output_prefix)

