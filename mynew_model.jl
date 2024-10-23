using Oceananigans
using Oceananigans: tupleit
using Oceananigans.Grids
using Oceananigans.Architectures: architecture
using Oceananigans.Advection: div_Uc
using Oceananigans.BoundaryConditions
using Oceananigans.Utils
using Oceananigans.Fields: TracerFields
using Oceananigans.Models.HydrostaticFreeSurfaceModels: PrescribedVelocityFields, tracernames, validate_tracer_advection, with_tracers, tracernames
using KernelAbstractions: @index, @kernel

import Oceananigans.Models: prognostic_fields

using Oceananigans.TimeSteppers: RungeKutta3TimeStepper, store_tendencies!, rk3_substep_field!
import Oceananigans.TimeSteppers: update_state!, time_step!, compute_tendencies!, rk3_substep!

mutable struct AdvectiveModel{G, A, V, T, B}
    grid :: G
    advection :: A
    velocities :: V
    tracers :: T
    timestepper :: B
end

struct DirectSpaceTimeAder end

function AdvectiveModel(; grid, 
                          velocities = PrescribedVelocityFields(; u = 1), 
                          tracers = :b, 
                          timestepper = nothing,
                          advection = WENO(; order = 5))

    # Next, we form a list of default boundary conditions:
    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)

    tracers = TracerFields(tracers, grid, NamedTuple())
    Gⁿ = deepcopy(tracers)
    G⁻ = deepcopy(tracers)
    
    if timestepper isa Nothing
        timestepper = DirectSpaceTime()
    else
        timestepper = RungeKutta3TimeStepper(grid, tracernames(tracers); Gⁿ = Gⁿ, G⁻ = G⁻)
    end

    model = AdvectiveModel(grid, advection, velocities, tracers, timestepper)
    update_state!(model)

    return model
end

prognostic_fields(model::AdvectiveModel) = model.tracers

update_state!(model::AdvectiveModel) = 
    fill_halo_regions!(model.tracers)

function time_step!(model::AdvectiveModel, Δt)
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and update state at iteration 0, in case run! is not used:
    update_state!(model)

    γ¹ = model.timestepper.γ¹
    γ² = model.timestepper.γ²
    γ³ = model.timestepper.γ³

    ζ² = model.timestepper.ζ²
    ζ³ = model.timestepper.ζ³

    compute_tendencies!(model)
    rk3_substep!(model, Δt, γ¹, nothing)
    store_tendencies!(model)
    update_state!(model)
    
    #
    # Second stage
    #

    compute_tendencies!(model)
    rk3_substep!(model, Δt, γ², ζ²)
    store_tendencies!(model)
    update_state!(model)

    #
    # Third stage
    #
    
    compute_tendencies!(model)
    rk3_substep!(model, Δt, γ³, ζ³)
    update_state!(model)

    return nothing
end

function rk3_substep!(model::AdvectiveModel, Δt, γ, ζ)
    grid       = model.grid
    arch       = architecture(grid)
    tracers    = model.tracers
    velocities = model.velocities

    for (tracer, Gⁿ, G⁻) in zip(tracers, model.timestepper.Gⁿ, model.timestepper.G⁻)
        launch!(arch, grid, :xyz, rk3_substep_field!, tracer, Δt, γ, ζ, Gⁿ, G⁻)
    end

    return nothing
end

function compute_tendencies!(model::AdvectiveModel) 
    grid       = model.grid
    arch       = architecture(grid)
    tendencies = model.timestepper.Gⁿ
    tracers    = model.tracers
    velocities = model.velocities

    for (tracer, G) in zip(tracers, tendencies)
        launch!(arch, grid, :xyz, _compute_tendency!, G, grid, model.advection, velocities, tracer)
    end

    return nothing
end

@kernel function _compute_tendency!(G, grid, advection, U, c)
    i, j, k = @index(Global, NTuple)
    @inbounds G[i, j, k] = - div_Uc(i, j, k, grid, advection, U, c)
end

function time_step!(model::AdvectiveModel{G, A, V, <:DirectSpaceTimeAder, B}, Δt) where {G, A, V, B}
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and update state at iteration 0, in case run! is not used:
    update_state!(model)

    compute_tendencies!(model)
    rk3_substep!(model, Δt, γ¹, nothing)
    store_tendencies!(model)
    update_state!(model)

    return nothing
end
