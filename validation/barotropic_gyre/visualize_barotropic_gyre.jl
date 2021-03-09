# # Barotropic gyre

using Oceananigans
using Oceananigans.Grids

using Oceananigans.Coriolis:
    HydrostaticSphericalCoriolis,
    VectorInvariantEnergyConserving,
    VectorInvariantEnstrophyConserving

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    HydrostaticFreeSurfaceModel,
    VectorInvariant,
    ExplicitFreeSurface

using Oceananigans.TurbulenceClosures: HorizontallyCurvilinearAnisotropicDiffusivity
using Oceananigans.Utils: prettytime, hours, day, days, years
using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval, IterationInterval

using Statistics
using JLD2
using Printf
using GLMakie

Nx = 360
Ny = 360

output_prefix = "barotropic_gyre_Nx$(Nx)_Ny$(Ny)"
filepath = output_prefix * ".jld2"

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

file = jldopen(filepath)

Nx = file["grid/Nx"]
Ny = file["grid/Ny"]

# A spherical domain
grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                    longitude = (-30, 30),
                                    latitude = (15, 75),
                                    z = (-4000, 0))

iterations = parse.(Int, keys(file["timeseries/t"]))

λu = xnodes(Face, grid)
φu = ynodes(Center, grid)

λc = xnodes(Center, grid)
φc = ynodes(Center, grid)

xu, yu, zu = geographic2cartesian(λu, φu)
xc, yc, zc = geographic2cartesian(λc, φc)

iter = Node(0)

plot_title = @lift @sprintf("Barotropic gyre: time = %s", prettytime(file["timeseries/t/" * string($iter)]))

u = @lift file["timeseries/u/" * string($iter)][:, :, 1]
η = @lift file["timeseries/η/" * string($iter)][:, :, 1]

fig = Figure(resolution = (2160, 1540))

ax = fig[1, 1] = LScene(fig)
wireframe!(ax, Sphere(Point3f0(0), 0.99f0), show_axis=false)
surface!(ax, xu, yu, zu, color=u, colormap=:balance)
rotate_cam!(ax.scene, (3π/4, -π/8, 0))
zoom!(ax.scene, (0, 0, 0), 2, true)

ax = fig[1, 2] = LScene(fig)
wireframe!(ax, Sphere(Point3f0(0), 0.99f0), show_axis=false)
surface!(ax, xc, yc, zc, color=η, colormap=:balance)
rotate_cam!(ax.scene, (3π/4, -π/8, 0))
zoom!(ax.scene, (0, 0, 0), 2, true)

supertitle = fig[0, :] = Label(fig, plot_title, textsize=50)

record(fig, output_prefix * ".mp4", iterations, framerate=30) do i
    @info "Animating iteration $i/$(iterations[end])..."
    iter[] = i
end

close(file)
