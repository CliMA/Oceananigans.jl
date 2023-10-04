using Oceananigans, Printf

using Oceananigans.Grids: φnode, λnode, halo_size
using Oceananigans.MultiRegion: getregion, number_of_regions
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_vector_halos!

Nx = 30
Ny = 30
Nz = 1

Lz = 1
R = 1 # sphere's radius
U = 1 # velocity scale
gravitational_acceleration = 100

grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz),
                                  z = (-Lz, 0),
                                  radius = R,
                                  horizontal_direction_halo = 4,
                                  partition = CubedSpherePartition(; R = 1))


# Solid body rotation
φʳ = 0        # Latitude pierced by the axis of rotation
α  = 90 - φʳ  # Angle between axis of rotation and north pole (degrees)
ψᵣ(λ, φ, z) = - U * R * (sind(φ) * cosd(α) - cosd(λ) * cosd(φ) * sind(α))

ψ = Field{Face, Face, Center}(grid)

# Here we avoid set! (which also isn't implemented btw) because we would like
# to manually determine the streamfunction within halo regions. This allows us
# to avoid having to fill_halo_regions correctly for a Face, Face, Center field.
for region in 1:number_of_regions(grid)
    i₀ = 1
    i⁺ = Nx + 1
    j₀ = 1
    j⁺ = Ny + 1
    k₀ = 1
    k⁺ = Nz + 1

    for k in k₀:k⁺, j=j₀:j⁺, i=i₀:i⁺
        λ = λnode(i, j, k, grid[region], Face(), Face(), Center())
        φ = φnode(i, j, k, grid[region], Face(), Face(), Center())
        ψ[region][i, j, k] = ψᵣ(λ, φ, 0)
    end
end

u = XFaceField(grid)
v = YFaceField(grid)

# What we want eventually:
# u .= - ∂y(ψ)
# v .= + ∂x(ψ)

for region in 1:number_of_regions(grid)
    u[region] .= - ∂y(ψ[region])
    v[region] .= + ∂x(ψ[region])
end

model = HydrostaticFreeSurfaceModel(; grid,
                                    momentum_advection = VectorInvariant(),
                                    free_surface = ExplicitFreeSurface(; gravitational_acceleration),
                                    tracer_advection = WENO(order=5),
                                    tracers = :θ,
                                    buoyancy = nothing)

# Initial conditions

for region in 1:number_of_regions(grid)
    model.velocities.u[region] .= - ∂y(ψ[region])
    model.velocities.v[region] .= + ∂x(ψ[region])
end

θ₀ = 1
Δφ = 20
θᵢ(λ, φ, z) = θ₀ * cosd(4λ) * exp(-φ^2 / 2Δφ^2)

set!(model, θ = θᵢ)

# Estimate time-step from the minimum grid spacing and the barotropic surface wave speed
Δx = minimum_xspacing(grid)
Δy = minimum_yspacing(grid)

c = sqrt(Lz * gravitational_acceleration) # surface wave speed
Δt = 0.1 * min(Δx, Δy) / c # CFL for free surface wave propagation

stop_time = 2π * U / R
simulation = Simulation(model; Δt, stop_time)

# Print a progress message

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(100))

tracer_fields = Field[]
save_tracer(sim) = push!(tracer_fields, deepcopy(sim.model.tracers.θ))
simulation.callbacks[:save_tracer] = Callback(save_tracer, IterationInterval(50))

run!(simulation)

@info "Making an animation from the saved data..."

# install Imaginocean.jl from GitHub
# using Pkg; Pkg.add(url="https://github.com/navidcy/Imaginocean.jl", rev="main")
using Imaginocean

using GLMakie, GeoMakie

n = Observable(1)

Θₙ = @lift tracer_fields[$n]

fig = Figure(resolution = (1600, 1200), fontsize=30)
ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
heatlatlon!(ax, Θₙ, colorrange=(-θ₀, θ₀), colormap = :balance)

fig

frames = 1:length(tracer_fields)

GLMakie.record(fig, "cubed_sphere_momentum_dynamics.mp4", frames, framerate = 12) do i
    @info string("Plotting frame ", i, " of ", frames[end])
    Θₙ[] = tracer_fields[i]
    heatlatlon!(ax, Θₙ, colorrange=(-θ₀, θ₀), colormap = :balance)
end
