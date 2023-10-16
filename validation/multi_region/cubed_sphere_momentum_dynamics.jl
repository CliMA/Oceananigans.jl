using Oceananigans, Printf

using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_vector_halos!
using Oceananigans.Grids: φnode, λnode, halo_size
using Oceananigans.MultiRegion: getregion, number_of_regions
using Oceananigans.Operators: ζ₃ᶠᶠᶜ

using Oceananigans.Utils
using KernelAbstractions: @kernel, @index

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
    i₀ = -3
    i⁺ = Nx + 4
    j₀ = -3
    j⁺ = Ny + 4
    k₀ = -3
    k⁺ = Nz + 4

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

for passes in 1:3
    fill_halo_regions!(u)
    fill_halo_regions!(v)
    @apply_regionally replace_horizontal_vector_halos!((; u, v, w = nothing), grid)
end



model = HydrostaticFreeSurfaceModel(; grid,
                                    momentum_advection = VectorInvariant(),
                                    # momentum_advection = nothing,
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
stop_time = 5Δt
simulation = Simulation(model; Δt, stop_time)

# Print a progress message
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|u|): %.2e, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.u),
                                prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))

tracer_fields = Field[]
save_tracer(sim) = push!(tracer_fields, deepcopy(sim.model.tracers.θ))

u_fields = Field[]
save_u(sim) = push!(u_fields, deepcopy(sim.model.velocities.u))

v_fields = Field[]
save_v(sim) = push!(v_fields, deepcopy(sim.model.velocities.v))

ζ = Field{Face, Face, Center}(grid)

vorticity_fields = Field[]

@kernel function _compute_vorticity!(ζ, grid, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds ζ[i, j, k] = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)
end

function save_vorticity(sim)
    Hx, Hy, Hz = halo_size(grid)

    u, v = sim.model.velocities.u, sim.model.velocities.v

    for passes in 1:3
        fill_halo_regions!(u)
        fill_halo_regions!(v)
        @apply_regionally replace_horizontal_vector_halos!((; u, v, w = nothing), grid)
    end

    for region in [1, 3, 5]
        for k in -Hz+1:Nz+Hz
            u[region][Nx+1, Ny+1:Ny+Hy, k] .= u[region][Nx+1:Nx+Hx, Ny+1, k]'
            v[region][Nx+1, Ny+1:Ny+Hy, k] .= v[region][Nx+1:Nx+Hx, Ny+1, k]'
        end
    end
    
    @apply_regionally begin
        params = KernelParameters(size(ζ) .+ 2 .* halo_size(grid), -1 .* halo_size(grid))
        launch!(architecture(grid), grid, params, _compute_vorticity!, ζ, grid, u, v)
    end

    push!(vorticity_fields, deepcopy(ζ))
end

save_fields_iteration_interval = 10
simulation.callbacks[:save_u] = Callback(save_u, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_v] = Callback(save_v, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_tracer] = Callback(save_tracer, IterationInterval(save_fields_iteration_interval))
simulation.callbacks[:save_vorticity] = Callback(save_vorticity, IterationInterval(save_fields_iteration_interval))

run!(simulation)

@info "Making an animation from the saved data..."

# install Imaginocean.jl from GitHub
# using Pkg; Pkg.add(url="https://github.com/navidcy/Imaginocean.jl", rev="main")
# using Imaginocean

using GLMakie, GeoMakie

n = Observable(1)

ζₙ = []
for region in 1:6
    push!(ζₙ, @lift parent(vorticity_fields[$n][region].data[:, :, 1]))
end

fig = Figure(resolution = (1600, 1200), fontsize=30)

axs = []
push!(axs, Axis(fig[3, 1]))
push!(axs, Axis(fig[3, 2]))
push!(axs, Axis(fig[2, 2]))
push!(axs, Axis(fig[2, 3]))
push!(axs, Axis(fig[1, 3]))
push!(axs, Axis(fig[1, 4]))

for region in 1:6
    heatmap!(axs[region], ζₙ[region], colorrange=(-2, 2), colormap = :balance)
end

fig

save("vorticity_test_2.png", fig)

frames = 1:length(vorticity_fields)

GLMakie.record(fig, "cubed_sphere_momentum_dynamics_vort.mp4", frames, framerate = 12) do i
    @info string("Plotting frame ", i, " of ", frames[end])

    ζₙ = []
    for region in 1:6
        push!(ζₙ, parent(vorticity_fields[i][region].data[:, :, 1]))
    end

    for region in 1:6
        heatmap!(axs[region], ζₙ[region], colorrange=(-2, 2), colormap = :balance)
    end
end


fig = Figure(resolution = (1600, 1200), fontsize=30)

axs = []
push!(axs, Axis(fig[3, 1]))
push!(axs, Axis(fig[3, 2]))
push!(axs, Axis(fig[2, 2]))
push!(axs, Axis(fig[2, 3]))
push!(axs, Axis(fig[1, 3]))
push!(axs, Axis(fig[1, 4]))

for region in 1:6
    heatmap!(axs[region], ζₙ[region], colorrange=(-2, 2), colormap = :balance)
end

fig

save("vorticity_test.png", fig)
