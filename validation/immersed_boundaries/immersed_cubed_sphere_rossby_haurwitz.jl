using Statistics
using Logging
using Printf
using DataDeps
using JLD2

using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary

using Oceananigans.Diagnostics: accurate_cell_advection_timescale

Logging.global_logger(OceananigansLogger())

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

dd32 = DataDep("cubed_sphere_32_grid",
    "Conformal cubed sphere grid with 32×32 grid points on each face",
    "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cubed_sphere_32_grid.jld2",
    "b1dafe4f9142c59a2166458a2def743cd45b20a4ed3a1ae84ad3a530e1eff538" # sha256sum
)

dd96 = DataDep("cubed_sphere_96_grid",
    "Conformal cubed sphere grid with 96×96 grid points on each face",
    "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cs96/cubed_sphere_96_grid.jld2"
)

DataDeps.register(dd32)
DataDeps.register(dd96)

function diagnose_velocities_from_streamfunction(ψ, grid)
    ψᶠᶠᶜ = Field(Face, Face,   Center, CPU(), grid)
    uᶠᶜᶜ = Field(Face, Center, Center, CPU(), grid)
    vᶜᶠᶜ = Field(Center, Face, Center, CPU(), grid)

    for (f, grid_face) in enumerate(grid.faces)
        Nx, Ny, Nz = size(grid_face)

        ψᶠᶠᶜ_face = ψᶠᶠᶜ.data.faces[f]
        uᶠᶜᶜ_face = uᶠᶜᶜ.data.faces[f]
        vᶜᶠᶜ_face = vᶜᶠᶜ.data.faces[f]

        for i in 1:Nx+1, j in 1:Ny+1
            ψᶠᶠᶜ_face[i, j, 1] = ψ(grid_face.λᶠᶠᵃ[i, j], grid_face.φᶠᶠᵃ[i, j])
        end

        for i in 1:Nx+1, j in 1:Ny
            uᶠᶜᶜ_face[i, j, 1] = (ψᶠᶠᶜ_face[i, j, 1] - ψᶠᶠᶜ_face[i, j+1, 1]) / grid.faces[f].Δyᶠᶜᵃ[i, j]
        end

        for i in 1:Nx, j in 1:Ny+1
            vᶜᶠᶜ_face[i, j, 1] = (ψᶠᶠᶜ_face[i+1, j, 1] - ψᶠᶠᶜ_face[i, j, 1]) / grid.faces[f].Δxᶜᶠᵃ[i, j]
        end
    end

    return uᶠᶜᶜ, vᶜᶠᶜ, ψᶠᶠᶜ
end

function cubed_sphere_rossby_haurwitz(grid_filepath; check_fields=false, nsteps=nothing, immersed=false, momvi=false)

    ## Grid setup

    H = 8kilometers
    underlying_grid = ConformalCubedSphereGrid(grid_filepath, Nz=1, z=(-H, 0))

    solid(x, y, z) = false

    if immersed
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(solid))
    else
        grid = underlying_grid
    end

    ## Model setup

    if momvi
        momentum_advection = VectorInvariant()
    else
        momentum_advection = nothing
    end

    model = HydrostaticFreeSurfaceModel(
                      grid = grid,
        momentum_advection = momentum_advection,
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
    ηᵢ(λ, ϕ)    = h(rescale²(ϕ), rescale¹(λ))

    #=
    #TODO: get this to work.
    ψ₀(λ, φ, z) = ψ(rescale²(φ), rescale¹(λ))

    u, v, w = model.velocities
    ψ = Field(Face, Face, Center, model.architecture, grid)
    u .= - ∂y(ψ)
    v .= + ∂x(ψ)

    # Note: this does _not_ fill halos!
    for i = 1:6
        face_ψ = Field(Face, Face, Center, model.architecture, get_face(grid, i))
        set!(face_ψ, ψ₀)

        face_u = get_face(model.velocities.u, i)
        face_v = get_face(model.velocities.v, i)

        face_u .= - ∂y(face_ψ)
        face_v .= + ∂x(face_ψ)
    end
    =#

    ψ₀(λ, φ) = ψ(rescale²(φ), rescale¹(λ))
    u₀, v₀, _ = diagnose_velocities_from_streamfunction(ψ₀, grid)

    # _Now_ we fill halos:
    Oceananigans.set!(model, u=u₀, v=v₀, η=ηᵢ)

    ## Simulation setup

    # Compute amount of time needed for a 45° rotation.
    angular_velocity = (n * (3+n) * ω - 2Ω) / ((1+n) * (2+n))
    stop_time = deg2rad(360) / abs(angular_velocity)
    Δt = 20seconds
    if nsteps != nothing
        stop_time=nsteps*Δt
    end
    @info "Stop time = $(prettytime(stop_time))"

    gravity_wave_speed = sqrt(g * H)
    min_spacing = filter(!iszero, grid.faces[1].Δyᶠᶠᵃ) |> minimum
    wave_propagation_time_scale = min_spacing / gravity_wave_speed
    gravity_wave_cfl = Δt / wave_propagation_time_scale
    @info "Gravity wave CFL = $gravity_wave_cfl"

    if !isnothing(model.closure)
        ν = model.closure.νh
        diffusive_cfl = ν * Δt / min_spacing^2
        @info "Diffusive CFL = $diffusive_cfl"
    end

    cfl = CFL(Δt, accurate_cell_advection_timescale)

    simulation = Simulation(model,
                        Δt = Δt,
                 stop_time = stop_time,
        iteration_interval = 20,
                  progress = Progress(time_ns()),
                parameters = (; cfl)
    )

    if check_fields
        fields_to_check = (
            u = model.velocities.u,
            v = model.velocities.v,
            η = model.free_surface.η,
           Gu = model.timestepper.Gⁿ.u,
           Gv = model.timestepper.Gⁿ.v,
           Gη = model.timestepper.Gⁿ.η
        )

        simulation.diagnostics[:state_checker] =
            StateChecker(model, fields=fields_to_check, schedule=IterationInterval(1))
    end

    output_fields = merge(model.velocities, (η=model.free_surface.η,))

    simulation.output_writers[:fields] =
    JLD2OutputWriter(model, output_fields,
        schedule = TimeInterval(1hour),
          prefix = "cubed_sphere_rossby_haurwitz",
           overwrite_existing = true)

    run!(simulation)

    return simulation
end


include("animate_on_map_projection.jl")

function run_cubed_sphere_rossby_haurwitz_validation(grid_filepath=datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2")

    simulation = cubed_sphere_rossby_haurwitz(grid_filepath)

    projections = [
        ccrs.NearsidePerspective(central_longitude=0, central_latitude=30),
        ccrs.NearsidePerspective(central_longitude=180, central_latitude=-30)
    ]

    animate_rossby_haurwitz(projections=projections)

    return simulation
end

grid_filepath=datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"
s1 = cubed_sphere_rossby_haurwitz(grid_filepath,check_fields=true,nsteps=1,immersed=true,momvi=true)
s2 = cubed_sphere_rossby_haurwitz(grid_filepath,check_fields=true,nsteps=1,immersed=false,momvi=true)
display([ s1.model.timestepper.Gⁿ.v.data[1][:,1,1]  s2.model.timestepper.Gⁿ.v.data[1][:,1,1] ] )
