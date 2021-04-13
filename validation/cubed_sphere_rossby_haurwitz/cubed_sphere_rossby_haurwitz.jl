using Statistics
using Logging
using Printf
using DataDeps
using JLD2

using Oceananigans
using Oceananigans.Units

using Oceananigans.Diagnostics: accurate_cell_advection_timescale

#####
##### Progress monitor
#####

mutable struct Progress
    interval_start_time :: Float64
end

function (p::Progress)(sim)
    wall_time = (time_ns() - p.interval_start_time) * 1e-9
    progress = 100 * (sim.model.clock.time / sim.stop_time)

    @info @sprintf("[%06.2f%%] Time: %s, iteration: %d, max(|u⃗|): (%.2e, %.2e) m/s, extrema(η): (min=%.2e, max=%.2e), CFL: %.2e, Δ(wall time): %s",
                   progress,
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   maximum(abs, sim.model.velocities.u),
                   maximum(abs, sim.model.velocities.v),
                   minimum(sim.model.free_surface.η),
                   maximum(sim.model.free_surface.η),
                   sim.parameters.cfl(sim.model),
                   prettytime(wall_time))

    p.interval_start_time = time_ns()

    return nothing
end

#####
##### Script starts here
#####

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

Logging.global_logger(OceananigansLogger())

dd = DataDep("cubed_sphere_32_grid",
    "Conformal cubed sphere grid with 32×32 grid points on each face",
    "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cubed_sphere_32_grid.jld2",
    "b1dafe4f9142c59a2166458a2def743cd45b20a4ed3a1ae84ad3a530e1eff538" # sha256sum
)

DataDeps.register(dd)

cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"

function diagnose_velocities_from_streamfunction(ψ, grid)
    ψᶠᶠᶜ = Field(Face, Face,   Center, CPU(), grid, nothing, nothing)
    uᶠᶜᶜ = Field(Face, Center, Center, CPU(), grid, nothing, nothing)
    vᶜᶠᶜ = Field(Center, Face, Center, CPU(), grid, nothing, nothing)

    for (f, grid_face) in enumerate(grid.faces)
        Nx, Ny, Nz = size(grid_face)
        for i in 1:Nx+1, j in 1:Ny+1
            ψᶠᶠᶜ.faces[f][i, j, 1] = ψ(grid_face.λᶠᶠᵃ[i, j], grid_face.φᶠᶠᵃ[i, j])
        end

        for i in 1:Nx+1, j in 1:Ny
            uᶠᶜᶜ.faces[f][i, j, 1] = (ψᶠᶠᶜ.faces[f][i, j, 1] - ψᶠᶠᶜ.faces[f][i, j+1, 1]) / grid.faces[f].Δyᶠᶜᵃ[i, j]
        end

        for i in 1:Nx, j in 1:Ny+1
            vᶜᶠᶜ.faces[f][i, j, 1] = (ψᶠᶠᶜ.faces[f][i+1, j, 1] - ψᶠᶠᶜ.faces[f][i, j, 1]) / grid.faces[f].Δxᶜᶠᵃ[i, j]
        end
    end

    return uᶠᶜᶜ, vᶜᶠᶜ, ψᶠᶠᶜ
end

function cubed_sphere_rossby_haurwitz()

    ## Grid setup

    H = 8kilometers
    grid = ConformalCubedSphereGrid(cs32_filepath, Nz=1, z=(-H, 0))

    ## Model setup

    model = HydrostaticFreeSurfaceModel(
              architecture = CPU(),
                      grid = grid,
        momentum_advection = VectorInvariant(),
        # momentum_advection = nothing,
              free_surface = ExplicitFreeSurface(gravitational_acceleration=100),
                  coriolis = HydrostaticSphericalCoriolis(scheme = VectorInvariantEnstrophyConserving()),
                   closure = nothing,
                   tracers = nothing,
                  buoyancy = nothing
    )

    ## Rossby-Haurwitz initial condition from Williamson et al. (§3.6, 1992)
    ## # here: θ ∈ [-π/2, π/2] is latitude, ϕ ∈ [0, 2π) is longitude

    R = grid.faces[1].radius
    K = 7.848e-6
    ω = 0
    n = 4

    g = model.free_surface.gravitational_acceleration
    Ω = model.coriolis.rotation_rate

    A(θ) = ω/2 * (2 * Ω + ω) * cos(θ)^2 + 1/4 * K^2 * cos(θ)^(2*n) * ((n+1) * cos(θ)^2 + (2 * n^2 - n - 2) - 2 * n^2 * sec(θ)^2 )
    B(θ) = 2 * K * (Ω + ω) * ((n+1) * (n+2))^(-1) * cos(θ)^(n) * ( n^2 + 2*n + 2 - (n+1)^2 * cos(θ)^2) # why not  (n+1)^2 sin(θ)^2 + 1
    C(θ)  = 1/4 * K^2 * cos(θ)^(2 * n) * ( (n+1) * cos(θ)^2 - (n+2))

    ψ(θ, ϕ) = -R^2 * ω * sin(θ)^2 + R^2 * K * cos(θ)^n * sin(θ) * cos(n*ϕ)

    u(θ, ϕ) =  R * ω * cos(θ) + R * K * cos(θ)^(n-1) * (n * sin(θ)^2 - cos(θ)^2) * cos(n*ϕ)
    v(θ, ϕ) = -n * K * R * cos(θ)^(n-1) * sin(θ) * sin(n*ϕ)

    h(θ, ϕ) = H + R^2/g * (A(θ) + B(θ) * cos(n * ϕ) + C(θ) * cos(2n * ϕ))

    # Total initial conditions
    # Previously: θ ∈ [-π/2, π/2] is latitude, ϕ ∈ [0, 2π) is longitude
    # Oceananigans: ϕ ∈ [-90, 90], λ ∈ [-180, 180],

    rescale¹(λ) = (λ + 180)/ 360 * 2π # λ to θ
    rescale²(ϕ) = ϕ / 180 * π # θ to ϕ

    # arguments were u(θ, ϕ), λ |-> ϕ, θ |-> ϕ
    uᵢ(λ, ϕ, z) = u(rescale²(ϕ), rescale¹(λ))
    vᵢ(λ, ϕ, z) = v(rescale²(ϕ), rescale¹(λ))
    ηᵢ(λ, ϕ, z) = h(rescale²(ϕ), rescale¹(λ))

    # set!(model, u=uᵢ, v=vᵢ, η = ηᵢ)

    ψ₀(λ, φ) = ψ(rescale²(φ), rescale¹(λ))

    u₀, v₀, _ = diagnose_velocities_from_streamfunction(ψ₀, grid)

    set!(model, u=u₀, v=v₀, η=ηᵢ)

    ## Simulation setup

    # Compute amount of time needed for a 360° (2π) rotation.
    angular_velocity = (n * (3+n) * ω - 2Ω) / ((1+n) * (2+n))
    stop_time = 2π / abs(angular_velocity)
    @info "Stop time = $(prettytime(stop_time))"

    Δt = 20seconds

    gravity_wave_speed = sqrt(g * H)
    min_spacing = filter(!iszero, grid.faces[1].Δyᶠᶠᵃ) |> minimum
    wave_propagation_time_scale = min_spacing / gravity_wave_speed
    gravity_wave_cfl = Δt / wave_propagation_time_scale
    @info "Gravity wave CFL = $gravity_wave_cfl"

    cfl = CFL(Δt, accurate_cell_advection_timescale)

    simulation = Simulation(model,
                        Δt = Δt,
                 stop_time = stop_time,
        iteration_interval = 20,
                  progress = Progress(time_ns()),
                parameters = (; cfl)
    )

    fields_to_check = (
        u = model.velocities.u,
        v = model.velocities.v,
        η = model.free_surface.η
    )

    simulation.diagnostics[:state_checker] =
        StateChecker(model, fields=fields_to_check, schedule=IterationInterval(20))

    output_fields = merge(model.velocities, (η=model.free_surface.η,))

    simulation.output_writers[:fields] =
    JLD2OutputWriter(model, output_fields,
        schedule = TimeInterval(5minutes),
          prefix = "cubed_sphere_rossby_haurwitz",
           force = true)

    run!(simulation)

    return simulation
end
