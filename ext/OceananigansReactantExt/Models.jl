module Models

using ..Grids: ShardedDistributed, ShardedGrid
using Oceananigans: Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel
using Oceananigans.TimeSteppers: RungeKutta3TimeStepper, SplitRungeKuttaTimeStepper, reconcile_state!
using Oceananigans.Architectures: architecture, convert_to_device
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Utils: configure_kernel, possibly_load_active_cells_map
using Reactant: Reactant, @trace

const SplitExplicitFreeSurfaces = Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces

const ReactantHFSM{TS, E} = Union{
    HydrostaticFreeSurfaceModel{TS, E, <:ReactantState},
    HydrostaticFreeSurfaceModel{TS, E, <:ShardedDistributed},
}

function Oceananigans.initialize!(model::ReactantHFSM)
    reconcile_state!(model)
    return nothing
end

# No-op for Reactant: the iteration == 0 check evaluates at trace time,
# causing a redundant update_state! to be compiled into every time_step!.
# Instead, first_time_step! handles initialization explicitly.
Oceananigans.TimeSteppers.maybe_prepare_first_time_step!(model::ReactantHFSM, Δt, callbacks) = nothing
Oceananigans.TimeSteppers.maybe_prepare_first_time_step!(model::ReactantHFSM{<:RungeKutta3TimeStepper}, Δt, callbacks) = nothing
Oceananigans.TimeSteppers.maybe_prepare_first_time_step!(model::ReactantHFSM{<:SplitRungeKuttaTimeStepper}, Δt, callbacks) = nothing

# Undo all the pipelining for a `ShardedDistributed` architecture
Oceananigans.Models.complete_communication_and_compute_buffer!(model, ::ShardedGrid, ::ShardedDistributed) = nothing
Oceananigans.Models.interior_tendency_kernel_parameters(::ShardedDistributed, grid) = :xyz

# Reactant uses CUDA version of the code to uplift program description to MLIR.
# Since default `weno_weight_computation` on CUDA uses LLVM's NVPTX intrinsics
# it causes Reactant to crash.
# We need to fall back to different optimization when running with Reactant
Oceananigans.Advection.default_weno_weight_computation(::ReactantState) = Oceananigans.Utils.ConvertingDivision{Float32}

#####
##### Split-explicit barotropic substeps as traced loops
#####

const SingleDeviceReactantGrid = AbstractGrid{<:Any, <:Any, <:Any, <:Any, <:ReactantState}

"""
    iterate_split_explicit_in_halo!(free_surface, grid::SingleDeviceReactantGrid, GUⁿ, GVⁿ, Δτᴮ, F, clock,
                                    weights, transport_weights, ::Val{Nsubsteps})

Advance the barotropic state through `Nsubsteps` substeps with a traced loop, so the compiled program holds
one copy of the substep kernels whatever `Nsubsteps` is. Single-device grids have no `Connected` sides, so the
substeps exchange no halos.
"""
function SplitExplicitFreeSurfaces.iterate_split_explicit_in_halo!(free_surface, grid::SingleDeviceReactantGrid,
                                                                   GUⁿ, GVⁿ, Δτᴮ, F::Fη, clock, weights,
                                                                   transport_weights, ::Val{Nsubsteps}) where {Fη, Nsubsteps}
    η           = free_surface.displacement
    grid        = free_surface.displacement.grid
    arch        = architecture(grid)
    state       = free_surface.filtered_state
    timestepper = free_surface.timestepper
    g           = free_surface.gravitational_acceleration
    U, V        = free_surface.barotropic_velocities

    # The traced loop has no halo exchange, which single-device grids never need
    interval = SplitExplicitFreeSurfaces.halo_exchange_interval(free_surface, grid, Nsubsteps)
    interval ≥ Nsubsteps || throw(ArgumentError("the traced substep loop exchanges no halos, but $(summary(grid)) " *
                                                "needs an exchange every $interval of $Nsubsteps substeps"))

    workspec = possibly_load_active_cells_map(grid, free_surface.kernel_parameters, false)
    barotropic_velocity_kernel!, _ = configure_kernel(arch, grid, workspec, SplitExplicitFreeSurfaces._split_explicit_barotropic_velocity!)
    free_surface_kernel!, _        = configure_kernel(arch, grid, workspec, SplitExplicitFreeSurfaces._split_explicit_free_surface!)

    U_args = convert_to_device(arch, (grid, Val(false), Δτᴮ, η, U, V, GUⁿ, GVⁿ, g, timestepper))
    η_args = convert_to_device(arch, (grid, Val(false), Δτᴮ, η, U, V, F, clock, state.η̅, state.U̅, state.V̅,
                                      state.Ũ, state.Ṽ, timestepper))

    # The substep counter is traced, so the weights are read from traced vectors
    traced_averaging_weights = Reactant.Ops.constant(collect(weights))
    traced_transport_weights = Reactant.Ops.constant(collect(transport_weights))

    @trace track_numbers=false for substep in 1:Nsubsteps
        barotropic_velocity_kernel!(U_args...)
        averaging_weight = Reactant.@allowscalar traced_averaging_weights[substep]
        transport_weight = Reactant.@allowscalar traced_transport_weights[substep]
        free_surface_kernel!(averaging_weight, transport_weight, η_args...)
    end

    return nothing
end

end # module
