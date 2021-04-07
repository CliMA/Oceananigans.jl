using Statistics
using Logging
using Printf
using DataDeps
using JLD2

using Oceananigans
using Oceananigans.Units
using Oceananigans.CubedSpheres
using Oceananigans.Coriolis
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.TurbulenceClosures

using Oceananigans.Diagnostics: accurate_cell_advection_timescale

Logging.global_logger(OceananigansLogger())

#####
##### state checker for debugging
#####

function state_checker(model)
    fields = model.tracers

    @info @sprintf("          |  minimum            maximum");
    for (name, field) in pairs(fields)
        for face_number in 1:length(model.grid.faces)
            min_val, max_val = field.faces[face_number] |> interior |> extrema
            @info @sprintf("%2s face %d | %+.12e %+.12e", name, face_number, min_val, max_val)
        end
        @info @sprintf("---------------------------------------------------")
    end

    return nothing
end

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

central_longitude = (0, 90,  0, 180, -90,   0)
central_latitude  = (0,  0, 90,   0,   0, -90)

"""
    tracer_advection_over_the_poles(; face_number, α)

Run a tracer advection experiment that initializes a cosine bell on face `face_number`
and advects it around the sphere over 12 days. `α` is the angle between the axis of
solid body rotation and the polar axis (degrees).
"""
function tracer_advection_over_the_poles(; face_number, α)

    grid = ConformalCubedSphereGrid(cs32_filepath, Nz=1, z=(-1, 0))

    ## Prescribed velocities and initial condition according to Williamson et al. (1992) §3.1

    period = 12days  # Time to make a full rotation (s)
    R = grid.faces[1].radius  # radius of the sphere (m)
    u₀ = 2π*R / perioid  # advecting velocity (m/s)

    # U(λ, φ, z) = u₀ * (cosd(φ) * cosd(α) + sind(φ) * cosd(λ) * sind(α))
    # V(λ, φ, z) = - u₀ * sind(λ) * sind(α)

    Ψ(λ, φ, z) = - R * u₀ * (sind(φ) * cosd(α) - cosd(λ) * cosd(φ) * sind(α))

    Ψᶠᶠᶜ = Field(Face, Face,   Center, CPU(), grid, nothing, nothing)
    Uᶠᶜᶜ = Field(Face, Center, Center, CPU(), grid, nothing, nothing)
    Vᶜᶠᶜ = Field(Center, Face, Center, CPU(), grid, nothing, nothing)
    Wᶜᶜᶠ = Field(Center, Center, Face, CPU(), grid, nothing, nothing)  # So we can use CFL

    for (f, grid_face) in enumerate(grid.faces)
        for i in 1:grid_face.Nx+1, j in 1:grid_face.Ny+1
            Ψᶠᶠᶜ.faces[f][i, j, 1] = Ψ(grid_face.λᶠᶠᵃ[i, j], grid_face.φᶠᶠᵃ[i, j], 0)
        end
    end

    for (f, grid_face) in enumerate(grid.faces)
        for i in 1:grid_face.Nx+1, j in 1:grid_face.Ny
            Uᶠᶜᶜ.faces[f][i, j, 1] = (Ψᶠᶠᶜ.faces[f][i, j, 1] - Ψᶠᶠᶜ.faces[f][i, j+1, 1]) / grid.faces[f].Δyᶠᶜᵃ[i, j]
        end
        for i in 1:grid_face.Nx, j in 1:grid_face.Ny+1
            Vᶜᶠᶜ.faces[f][i, j, 1] = (Ψᶠᶠᶜ.faces[f][i+1, j, 1] - Ψᶠᶠᶜ.faces[f][i, j, 1]) / grid.faces[f].Δxᶜᶠᵃ[i, j]
        end
    end

    ## Model setup

    model = HydrostaticFreeSurfaceModel(
        architecture = CPU(),
                grid = grid,
             tracers = :h,
          velocities = PrescribedVelocityFields(u=Uᶠᶜᶜ, v=Vᶜᶠᶜ, w=Wᶜᶜᶠ),
        free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1),
            coriolis = nothing,
             closure = nothing,
            buoyancy = nothing
    )

    ## Cosine bell initial condition according to Williamson et al. (1992) §3.1

    h₀ = 1000 # meters
    λ₀ = central_longitude[face_number]
    φ₀ = central_longitude[face_number]

    # Great circle distance between (λ, φ) and the center of the cosine bell (λ₀, φ₀)
    # using the haversine formula
    r(λ, φ) = 2R * asin(√(sind((φ - φ₀) / 2)^2 + cosd(φ) * cosd(φ₀) * sind((λ - λ₀) / 2)^2))

    cosine_bell(λ, φ, z) = r(λ, φ) < R ? h₀/2 * (1 + cos(π * r(λ, φ) / R)) : 0

    set!(model, h=cosine_bell)

    ## Simulation setup

    Δt = 10minutes

    cfl = CFL(Δt, accurate_cell_advection_timescale)

    simulation = Simulation(model,
                            Δt = Δt,
                            stop_time = 12days,
                            iteration_interval = 1,
                            progress = Progress(time_ns()),
                            parameters = (; cfl))

    # TODO: Implement NaNChecker for ConformalCubedSphereField
    empty!(simulation.diagnostics)

    outputs = (u=Uᶠᶜᶜ, v=Vᶜᶠᶜ, h=model.tracers.h)

    simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                        schedule = TimeInterval(1hour),
                                                        prefix = "tracer_advection_over_the_poles_face$face_number",
                                                        force = true)

    run!(simulation)

    return simulation
end

# for face_number in 1:6
#     tracer_advection_over_the_poles(face_number)
# end

# include("animate_on_map.jl")

# for face_number in 1:6
#     animate_tracer_advection(face_number)
# end
