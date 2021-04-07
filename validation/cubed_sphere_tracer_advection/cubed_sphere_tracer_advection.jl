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

using Oceananigans.CubedSpheres: ConformalCubedSphereFunctionField

import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.CubedSpheres: fill_horizontal_velocity_halos!

fill_halo_regions!(::ConformalCubedSphereFunctionField, args...) = nothing

# Forget about filling velocity halos when `velocities = PrescribedVelocityFields`
fill_horizontal_velocity_halos!(u::ConformalCubedSphereFunctionField, v, arch) = nothing
fill_horizontal_velocity_halos!(u, v::ConformalCubedSphereFunctionField, arch) = nothing
fill_horizontal_velocity_halos!(u::ConformalCubedSphereFunctionField, v::ConformalCubedSphereFunctionField, arch) = nothing

import Oceananigans.CubedSpheres: maybe_replace_with_face

maybe_replace_with_face(velocities::PrescribedVelocityFields, cubed_sphere_grid, face_number) =
    PrescribedVelocityFields(velocities.u.faces[face_number], velocities.v.faces[face_number], velocities.w.faces[face_number], velocities.parameters)

import Oceananigans.CubedSpheres: fill_horizontal_velocity_halos!

# We will use prescribed velocity `Field`s
fill_horizontal_velocity_halos!(u, v, arch) = nothing

#####
##### state checker for debugging
#####

# Takes forever to compile with Julia 1.6...
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

Logging.global_logger(OceananigansLogger())

dd = DataDep("cubed_sphere_32_grid",
    "Conformal cubed sphere grid with 32×32 grid points on each face",
    "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cubed_sphere_32_grid.jld2",
    "b1dafe4f9142c59a2166458a2def743cd45b20a4ed3a1ae84ad3a530e1eff538" # sha256sum
)

DataDeps.register(dd)

cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"

central_longitude = (0, 90, 0, 180, -90, 0)
central_latitude  = (0, 0, 90, 0, 0, -90)

function tracer_advection_over_the_poles(face_number)

    H = 4kilometers
    grid = ConformalCubedSphereGrid(cs32_filepath, Nz=1, z=(-H, 0))

    ## Prescribed velocities and initial condition according to Williamson et al. (1992) §3.1

    R = grid.faces[1].radius  # radius of the sphere (m)
    u₀ = 2π*R / (12days)  # advecting velocity (m/s)
    α = 0  # angle between the axis of solid body rotation and the polar axis (degrees)

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

    # fill_halo_regions!(Ψᶠᶠᶜ, CPU())
    # fill_halo_regions!(Ψᶠᶠᶜ, CPU()) # get those corners

    for (f, grid_face) in enumerate(grid.faces)
        for i in 1:grid_face.Nx+1, j in 1:grid_face.Ny
            Uᶠᶜᶜ.faces[f][i, j, 1] = (Ψᶠᶠᶜ.faces[f][i, j, 1] - Ψᶠᶠᶜ.faces[f][i, j+1, 1]) / grid.faces[f].Δyᶠᶜᵃ[i, j]
        end
        for i in 1:grid_face.Nx, j in 1:grid_face.Ny+1
            Vᶜᶠᶜ.faces[f][i, j, 1] = (Ψᶠᶠᶜ.faces[f][i+1, j, 1] - Ψᶠᶠᶜ.faces[f][i, j, 1]) / grid.faces[f].Δxᶜᶠᵃ[i, j]
        end
    end

    # fill_horizontal_velocity_halos!(Uᶠᶜᶜ, Vᶜᶠᶜ, CPU())
    # fill_horizontal_velocity_halos!(Uᶠᶜᶜ, Vᶜᶠᶜ, CPU()) # get those corners

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
    r(λ, φ) = R * acos(sind(φ₀) * sind(φ) + cosd(φ₀) * cosd(φ) * cosd(λ - λ₀))

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

for face_number in 1:6
    tracer_advection_over_the_poles(face_number)
end

include("animate_on_map.jl")

for face_number in 1:6
    animate_tracer_advection(face_number)
end
