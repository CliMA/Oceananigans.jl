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

#####
##### Gotta dispatch on some stuff after defining Oceananigans.CubedSpheres
#####

using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface

import Oceananigans.CubedSpheres: maybe_replace_with_face

maybe_replace_with_face(free_surface::ExplicitFreeSurface, cubed_sphere_grid, face_number) =
  ExplicitFreeSurface(free_surface.η.faces[face_number], free_surface.gravitational_acceleration)

import Oceananigans.Diagnostics: accurate_cell_advection_timescale

function accurate_cell_advection_timescale(grid::ConformalCubedSphereGrid, velocities)

    min_timescale_on_faces = []

    for (face_number, grid_face) in enumerate(grid.faces)
        velocities_face = maybe_replace_with_face(velocities, grid, face_number)
        min_timescale_on_face = accurate_cell_advection_timescale(grid_face, velocities_face)
        push!(min_timescale_on_faces, min_timescale_on_face)
    end

    return minimum(min_timescale_on_faces)
end

import Oceananigans.OutputWriters: fetch_output

fetch_output(field::ConformalCubedSphereField, model, field_slicer) =
    Tuple(fetch_output(field_face, model, field_slicer) for field_face in field.faces)

import Base: minimum, maximum

minimum(field::ConformalCubedSphereField; dims=:) = minimum(minimum(field_face; dims) for field_face in field.faces)
maximum(field::ConformalCubedSphereField; dims=:) = maximum(maximum(field_face; dims) for field_face in field.faces)

minimum(f, field::ConformalCubedSphereField; dims=:) = minimum(minimum(f, field_face; dims) for field_face in field.faces)
maximum(f, field::ConformalCubedSphereField; dims=:) = maximum(maximum(f, field_face; dims) for field_face in field.faces)

#####
##### state checker for debugging
#####

# Takes forever to compile with Julia 1.6...
function state_checker(model)
    fields = (
        u = model.velocities.u,
        v = model.velocities.v,
        w = model.velocities.w,
        η = model.free_surface.η,
        Gu = model.timestepper.Gⁿ.u,
        Gv = model.timestepper.Gⁿ.v,
        Gη = model.timestepper.Gⁿ.η
    )

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

    @info @sprintf("Time: %s, iteration: %d, max(u⃗): (%.2e, %.2e) m/s, extrema(η): (min=%.2e, max=%.2e), CFL: %.2e, wall time: %s",
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   maximum(abs, sim.model.velocities.u),
                   maximum(abs, sim.model.velocities.v),
                   minimum(abs, sim.model.free_surface.η),
                   maximum(abs, sim.model.free_surface.η),
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
    "3cc5d86290c3af028cddfa47e61e095ee470fe6f8d779c845de09da2f1abeb15" # sha256sum
)

DataDeps.register(dd)

cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"

central_longitude = (0, 90, 0, 180, -90, 0)
central_latitude  = (0, 0, 90, 0, 0, -90)

function surface_gravity_waves_on_cubed_sphere(face_number)

    H = 4kilometers
    grid = ConformalCubedSphereGrid(cs32_filepath, Nz=1, z=(-H, 0))

    ## Model setup

    model = HydrostaticFreeSurfaceModel(
              architecture = CPU(),
                      grid = grid,
        momentum_advection = VectorInvariant(),
              free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1),
            # free_surface = ImplicitFreeSurface(gravitational_acceleration=0.1)
                  coriolis = nothing,
                # coriolis = HydrostaticSphericalCoriolis(scheme = VectorInvariantEnstrophyConserving()),
                   closure = nothing,
                   tracers = nothing,
                  buoyancy = nothing
    )

    ## Initial condition:
    ## Very small sea surface height perturbation so the resulting dynamics are well-described
    ## by a linear free surface.

    A  = 1e-5 * H  # Amplitude of the perturbation
    λ₀ = central_longitude[face_number]
    φ₀ = central_latitude[face_number]
    Δλ = 15  # Longitudinal width
    Δφ = 15  # Latitudinal width

    Ξ(λ, φ, z) = 1e-5 * randn()
    η′(λ, φ, z) = A * exp(- (λ - λ₀)^2 / Δλ^2) * exp(- (φ - φ₀)^2 / Δφ^2)

    # set!(model, u=Ξ, v=Ξ, η=η′)
    set!(model, η=η′)

    ## Simulation setup

    Δt = 20minutes

    cfl = CFL(Δt, accurate_cell_advection_timescale)

    simulation = Simulation(model,
                            Δt = Δt,
                            stop_time = 5days,
                            iteration_interval = 1,
                            progress = Progress(time_ns()),
                            parameters = (; cfl))

    # TODO: Implement NaNChecker for ConformalCubedSphereField
    empty!(simulation.diagnostics)

    output_fields = merge(model.velocities, (η=model.free_surface.η,))

    simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                        schedule = TimeInterval(1hour),
                                                        prefix = "surface_gravity_waves_on_cubed_sphere_face$face_number",
                                                        force = true)

    run!(simulation)

    return simulation
end

include("animate_sphere_on_map.jl")

for face_number in 1:6
    surface_gravity_waves_on_cubed_sphere(face_number)
    animate_surface_gravity_waves_on_cubed_sphere(face_number)
end
