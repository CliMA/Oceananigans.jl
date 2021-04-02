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

H = 4kilometers
grid = ConformalCubedSphereGrid(cs32_filepath, Nz=1, z=(-H, 0))

## Prescribed velocities and initial condition according to Williamson et al. (1992) §3.1

R = grid.faces[1].radius  # radius of the sphere (m)
u₀ = 2π*R / (12days)  # advecting velocity (m/s)
α = 0  # angle between the axis of solid body rotation and the polar axis (degrees)

U(λ, φ, z) = u₀ * (cosd(φ) * cosd(α) + sind(φ) * cosd(λ) * sind(α))
V(λ, φ, z) = - u₀ * sind(λ) * sind(α)

U1(λ, φ, z) = U(λ, φ, z)
V1(λ, φ, z) = V(λ, φ, z)

U2(λ, φ, z) = U(λ, φ, z)
V2(λ, φ, z) = V(λ, φ, z)

U3(λ, φ, z) = +V(λ, φ, z)
V3(λ, φ, z) = -U(λ, φ, z)

U4(λ, φ, z) = -V(λ, φ, z)
V4(λ, φ, z) = +U(λ, φ, z)

U4(λ, φ, z) = -V(λ, φ, z)
V4(λ, φ, z) = +U(λ, φ, z)

U5(λ, φ, z) = -V(λ, φ, z)
V5(λ, φ, z) = +U(λ, φ, z)

U6(λ, φ, z) = U(λ, φ, z)
V6(λ, φ, z) = V(λ, φ, z)

zerofunc(args...) = 0

U_faces = (
    FunctionField{Face, Center, Center}(U1, grid.faces[1]),
    FunctionField{Face, Center, Center}(U2, grid.faces[2]),
    FunctionField{Face, Center, Center}(U3, grid.faces[3]),
    FunctionField{Face, Center, Center}(U4, grid.faces[4]),
    FunctionField{Face, Center, Center}(U5, grid.faces[5]),
    FunctionField{Face, Center, Center}(U6, grid.faces[6]),
)

V_faces = (
    FunctionField{Center, Face, Center}(V1, grid.faces[1]),
    FunctionField{Center, Face, Center}(V2, grid.faces[2]),
    FunctionField{Center, Face, Center}(V3, grid.faces[3]),
    FunctionField{Center, Face, Center}(V4, grid.faces[4]),
    FunctionField{Center, Face, Center}(V5, grid.faces[5]),
    FunctionField{Center, Face, Center}(V6, grid.faces[6]),
)

W_faces = (
    FunctionField{Center, Center, Face}(zerofunc, grid.faces[1]),
    FunctionField{Center, Center, Face}(zerofunc, grid.faces[2]),
    FunctionField{Center, Center, Face}(zerofunc, grid.faces[3]),
    FunctionField{Center, Center, Face}(zerofunc, grid.faces[4]),
    FunctionField{Center, Center, Face}(zerofunc, grid.faces[5]),
    FunctionField{Center, Center, Face}(zerofunc, grid.faces[6]),
)

U_ff = ConformalCubedSphereFunctionField{Face, Center, Center, typeof(U_faces), typeof(grid)}(U_faces)
V_ff = ConformalCubedSphereFunctionField{Center, Face, Center, typeof(V_faces), typeof(grid)}(V_faces)
W_ff = ConformalCubedSphereFunctionField{Center, Center, Face, typeof(W_faces), typeof(grid)}(W_faces)

velocities = PrescribedVelocityFields(U_ff, V_ff, W_ff, nothing)

## Model setup

model = HydrostaticFreeSurfaceModel(
    architecture = CPU(),
            grid = grid,
         tracers = :h,
      velocities = velocities,
    # velocities = PrescribedVelocityFields(u=U, v=V),
    free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1),
        coriolis = nothing,
         closure = nothing,
        buoyancy = nothing
)

## Cosine bell initial condition according to Williamson et al. (1992) §3.1

h₀ = 1000 # meters
λ₀ = 0    # Central longitude
φ₀ = 0    # Central latitude

# Great circle distance between (λ, φ) and the center of the cosine bell (λ₀, φ₀)
r(λ, φ) = R * acos(sind(φ₀) * sind(φ) + cosd(φ₀) * cosd(φ) * cosd(λ - λ₀))

cosine_bell(λ, φ, z) = r(λ, φ) < R ? h₀/2 * (1 + cos(π * r(λ, φ) / R)) : 0

set!(model, h=cosine_bell)

## Simulation setup

Δt = 20minutes

mutable struct Progress
    interval_start_time :: Float64
end

function (p::Progress)(sim)
    wall_time = (time_ns() - p.interval_start_time) * 1e-9

    @info @sprintf("Time: %s, iteration: %d, extrema(h): (min=%.2e, max=%.2e), wall time: %s",
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   minimum(abs, sim.model.tracers.h),
                   maximum(abs, sim.model.tracers.h),
                   prettytime(wall_time))

    p.interval_start_time = time_ns()

    return nothing
end

simulation = Simulation(model,
                        Δt = Δt,
                        stop_time = 1days,
                        iteration_interval = 1,
                        progress = Progress(time_ns()))

# TODO: Implement NaNChecker for ConformalCubedSphereField
empty!(simulation.diagnostics)

simulation.output_writers[:fields] = JLD2OutputWriter(model, model.tracers,
                                                      schedule = TimeInterval(1hour),
                                                      prefix = "tracer_advection_over_the_poles",
                                                      force = true)

run!(simulation)
