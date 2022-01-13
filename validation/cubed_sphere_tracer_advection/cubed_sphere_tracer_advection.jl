using Statistics
using Logging
using Printf
using DataDeps
using JLD2

using Oceananigans
using Oceananigans.Units

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

    @info @sprintf("Time: %s, iteration: %d, extrema(h): (min=%.2e, max=%.2e), CFL: %.2e, wall time: %s",
                prettytime(sim.model.clock.time),
                sim.model.clock.iteration,
                minimum(sim.model.tracers.h),
                maximum(sim.model.tracers.h),
                sim.parameters.cfl(sim.model),
                prettytime(wall_time))

    p.interval_start_time = time_ns()

    return nothing
end

#####
##### Script starts here
#####

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

dd = DataDep("cubed_sphere_32_grid",
    "Conformal cubed sphere grid with 32×32 grid points on each face",
    "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cubed_sphere_32_grid.jld2",
    "b1dafe4f9142c59a2166458a2def743cd45b20a4ed3a1ae84ad3a530e1eff538" # sha256sum
)

DataDeps.register(dd)

cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"

# Central (λ, φ) for each face of the cubed sphere
central_longitude = (0, 90,  0, 180, -90,   0)
central_latitude  = (0,  0, 90,   0,   0, -90)

"""
    cubed_sphere_tracer_advection(; face_number, α)

Run a tracer advection experiment that initializes a cosine bell on face `face_number`
and advects it around the sphere over 12 days. `α` is the angle between the axis of
solid body rotation and the polar axis (degrees).
"""
function cubed_sphere_tracer_advection(; face_number, α)

    grid = ConformalCubedSphereGrid(cs32_filepath, Nz=1, z=(-1, 0))

    ## Prescribed velocities and initial condition according to Williamson et al. (1992) §3.1

    period = 12days  # Time to make a full rotation (s)
    R = grid.faces[1].radius  # radius of the sphere (m)
    u₀ = 2π*R / period  # advecting velocity (m/s)

    # U(λ, φ, z) = u₀ * (cosd(φ) * cosd(α) + sind(φ) * cosd(λ) * sind(α))
    # V(λ, φ, z) = - u₀ * sind(λ) * sind(α)

    Ψ(λ, φ, z) = - R * u₀ * (sind(φ) * cosd(α) - cosd(λ) * cosd(φ) * sind(α))

    Ψᶠᶠᶜ = Field(Face, Face,   Center, CPU(), grid)
    Uᶠᶜᶜ = Field(Face, Center, Center, CPU(), grid)
    Vᶜᶠᶜ = Field(Center, Face, Center, CPU(), grid)
    Wᶜᶜᶠ = Field(Center, Center, Face, CPU(), grid)  # So we can use CFL

    for (f, grid_face) in enumerate(grid.faces)
        for i in 1:grid_face.Nx+1, j in 1:grid_face.Ny+1
            Ψᶠᶠᶜ.data.faces[f][i, j, 1] = Ψ(grid_face.λᶠᶠᵃ[i, j], grid_face.φᶠᶠᵃ[i, j], 0)
        end
    end

    for (f, grid_face) in enumerate(grid.faces)
        Ψᶠᶠᶜ_face = Ψᶠᶠᶜ.data.faces[f]
        Uᶠᶜᶜ_face = Uᶠᶜᶜ.data.faces[f]
        Vᶜᶠᶜ_face = Vᶜᶠᶜ.data.faces[f]

        for i in 1:grid_face.Nx+1, j in 1:grid_face.Ny
            Uᶠᶜᶜ_face[i, j, 1] = (Ψᶠᶠᶜ_face[i, j, 1] - Ψᶠᶠᶜ_face[i, j+1, 1]) / grid.faces[f].Δyᶠᶜᵃ[i, j]
        end

        for i in 1:grid_face.Nx, j in 1:grid_face.Ny+1
            Vᶜᶠᶜ_face[i, j, 1] = (Ψᶠᶠᶜ_face[i+1, j, 1] - Ψᶠᶠᶜ_face[i, j, 1]) / grid.faces[f].Δxᶜᶠᵃ[i, j]
        end
    end

    ## Model setup

    model = HydrostaticFreeSurfaceModel(
                      grid = grid,
        momentum_advection = nothing,
                   tracers = :h,
                velocities = PrescribedVelocityFields(u=Uᶠᶜᶜ, v=Vᶜᶠᶜ, w=Wᶜᶜᶠ),
              free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1),
                  coriolis = nothing,
                   closure = nothing,
                  buoyancy = nothing
    )

    ## Cosine bell initial condition according to Williamson et al. (1992) §3.1

    h₀ = 1000
    λ₀ = central_longitude[face_number]
    φ₀ = central_longitude[face_number]

    # Great circle distance between (λ, φ) and the center of the cosine bell (λ₀, φ₀)
    # using the haversine formula
    r(λ, φ) = 2R * asin(√(sind((φ - φ₀) / 2)^2 + cosd(φ) * cosd(φ₀) * sind((λ - λ₀) / 2)^2))

    cosine_bell(λ, φ, z) = r(λ, φ) < R ? h₀/2 * (1 + cos(π * r(λ, φ) / R)) : 0

    Oceananigans.set!(model, h=cosine_bell)

    ## Simulation setup

    Δt = 10minutes

    cfl = CFL(Δt, accurate_cell_advection_timescale)

    simulation = Simulation(model,
                        Δt = Δt,
                 stop_time = 1period,
        iteration_interval = 1,
                  progress = Progress(time_ns()),
                parameters = (; cfl)
    )

    outputs = (u=Uᶠᶜᶜ, v=Vᶜᶠᶜ, h=model.tracers.h)

    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, outputs,
            schedule = TimeInterval(1hour),
              prefix = "cubed_sphere_tracer_advection_face$(face_number)_alpha$α",
              force = true)

    run!(simulation)

    return simulation
end

#####
##### Run all the experiments!
#####

include("animate_on_map_projection.jl")

function run_cubed_sphere_tracer_advection_validation()

    αs = (0, 45, 90)

    for f in 1:6, α in αs
        cubed_sphere_tracer_advection(face_number=f, α=α)
    end

    projections = [
        ccrs.NearsidePerspective(central_longitude=0,   central_latitude=30),
        ccrs.NearsidePerspective(central_longitude=180, central_latitude=-30)
    ]

    for f in 1:6, α in αs
        animate_tracer_advection(face_number=f, α=α, projections=projections)
    end
end
