using Oceananigans: fields
using Oceananigans.DistributedComputations: maybe_distributed_fill_halo_regions!
using KernelAbstractions.Extras.LoopInfo: @unroll

# Include buffers for distributed grids
@inline build_halo_fill_args(f, grid, args...) = (f.data, f.boundary_conditions, f.indices, instantiated_location(f), grid, args...)
@inline build_halo_fill_args(f, grid::DistributedGrid, args...) = (f.data, f.boundary_conditions, f.indices, instantiated_location(f), grid, f.communication_buffers, args...)

# `CompleteHaloFilling` communicates every substep and needs the field's real communication buffers,
# which `convert_to_device` strips to `nothing` on a GPU. Leave its distributed args unconverted.
@inline prepare_halo_fill_args(arch, args, grid, free_surface) = convert_to_device(arch, args)
@inline prepare_halo_fill_args(arch, args, grid::DistributedGrid, ::SplitExplicitFreeSurface{CompleteHaloFilling}) = args

# Selection between topology-aware and non-aware operators depending on
# whether we fill halos or not in between substeps.
#
# filled_halos = Val(false): halos are NOT filled each substep (extend_halos mode).
#   → Use topology-aware operators because halo data goes stale after the first substep.
#
# filled_halos = Val(true): halos ARE filled each substep (fill_halos mode).
#   → Use non-topology-aware operators because halo data is always fresh.
@inline x_derivative_operator(::Val{false}) = ∂xᵣTᶠᶜᶠ
@inline x_derivative_operator(::Val{true})  = ∂xᵣᶠᶜᶠ
@inline y_derivative_operator(::Val{false}) = ∂yᵣTᶜᶠᶠ
@inline y_derivative_operator(::Val{true})  = ∂yᵣᶜᶠᶠ

@inline x_difference_operator(::Val{false}) = δxTᶜᵃᵃ
@inline x_difference_operator(::Val{true})  = δxᶜᶜᶜ
@inline y_difference_operator(::Val{false}) = δyTᵃᶜᵃ
@inline y_difference_operator(::Val{true})  = δyᶜᶜᶜ

@inline x_column_depth(i, j, k, grid, ::Val{false}, η) = column_depthTᶠᶜᵃ(i, j, k, grid, η)
@inline x_column_depth(i, j, k, grid, ::Val{true},  η) =  column_depthᶠᶜᵃ(i, j, k, grid, η)
@inline y_column_depth(i, j, k, grid, ::Val{false}, η) = column_depthTᶜᶠᵃ(i, j, k, grid, η)
@inline y_column_depth(i, j, k, grid, ::Val{true},  η) =  column_depthᶜᶠᵃ(i, j, k, grid, η)

# Evolution Kernels
#
# ∂t(η) = - ∇⋅U
# ∂t(U) = - gH∇η + f
#
# The free surface field η and its average η̄ are located on `Face`s at the surface (grid.Nz +1). All other intermediate
# variables (U, V, Ū, V̄) are barotropic fields (`ReducedField`) for which a k index is not defined.
@kernel function _split_explicit_barotropic_velocity!(averaging_weight, sᵐ, grid, filled_halos, Δτ, η, U, V,
                                                      Gᵁ, Gⱽ, Gᵁᶜ, Gⱽᶜ, w, g, U̅, V̅, timestepper)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    cache_previous_velocities!(timestepper, i, j, 1, U, V)

    Hᶠᶜ = x_column_depth(i, j, k_top, grid, filled_halos, η) # topology-aware column
    Hᶜᶠ = y_column_depth(i, j, k_top, grid, filled_halos, η) # topology-aware column
    ∂xᵣ = x_derivative_operator(filled_halos)
    ∂yᵣ = y_derivative_operator(filled_halos)

    # With the frozen forcing this dispatches back to a plain `Gᵁ[i, j, 1]` load.
    Gᵁˢ = slow_forcing(i, j, Gᵁ, Gᵁᶜ, w, sᵐ)
    Gⱽˢ = slow_forcing(i, j, Gⱽ, Gⱽᶜ, w, sᵐ)

    # ∂τ(U) = - ∇η + G, using ∂xᵣT/∂yᵣT (derivatives at constant r) since η lives on the surface and has
    # no vertical structure.
    @inbounds begin
        U[i, j, 1] += Δτ * (- g * Hᶠᶜ * ∂xᵣ(i, j, k_top, grid, η★, timestepper, η) + Gᵁˢ)
        V[i, j, 1] += Δτ * (- g * Hᶜᶠ * ∂yᵣ(i, j, k_top, grid, η★, timestepper, η) + Gⱽˢ)

        # Time-averaging the barotropic velocity
        U̅[i, j, 1] += averaging_weight * U[i, j, 1]
        V̅[i, j, 1] += averaging_weight * V[i, j, 1]
    end
end

@kernel function _split_explicit_free_surface!(averaging_weight, transport_weight, grid, filled_halos, Δτ, η, U, V, F, clock, η̅, Ũ, Ṽ, timestepper)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    cache_previous_free_surface!(timestepper, i, j, k_top, η)

    δx = x_difference_operator(filled_halos)
    δy = y_difference_operator(filled_halos)

    δh_U = (δx(i, j, grid.Nz, grid, Δy_qᶠᶜᶠ, U★, timestepper, U) +
            δy(i, j, grid.Nz, grid, Δx_qᶜᶠᶠ, V★, timestepper, V)) * Az⁻¹ᶜᶜᶠ(i, j, k_top, grid)

    @inbounds begin
        η[i, j, k_top] += Δτ * (F(i, j, k_top, grid, clock, (; η, U, V)) - δh_U)

        # Time-averaging η and the transport U★/V★ that advanced it, which is what constancy needs.
        η̅[i, j, k_top] += averaging_weight * η[i, j, k_top]
        Ũ[i, j, 1]     += transport_weight * U★(i, j, 1, grid, timestepper, U)
        Ṽ[i, j, 1]     += transport_weight * V★(i, j, 1, grid, timestepper, V)
    end
end

#####
##### Multi-stage substep kernel (RungeKutta3Scheme)
#####
##### Each barotropic substep runs three RK stages from the substep-start state (ηⁿ, Uⁿ, Vⁿ). Free surface and
##### velocity commute within a stage, since both depend only on the previous-stage state, so one kernel does
##### both; the previous stage needs its own buffer to keep the δx stencil from racing on the live fields.
##### The averages (η̅, U̅, V̅, Ũ, Ṽ) accumulate only on the final stage, which is what preserves constancy.

# The state triples travel as tuples: `isregional` recurses element by element over the argument list and
# stops inferring on a long one.
@kernel function _barotropic_stage!(averaging_weight, transport_weight, sᵐ, Δτ, grid, filled_halos,
                                    substep_state, previous_state, stage_state, slow_forcings, g, F, clock, averages)

    i, j = @index(Global, NTuple)
    k_top = grid.Nz + 1

    ηⁿ, Uⁿ, Vⁿ = substep_state
    ηᵖ, Uᵖ, Vᵖ = previous_state
    η,  U,  V  = stage_state
    Gᵁ, Gⱽ, Gᵁᶜ, Gⱽᶜ, w = slow_forcings
    η̅, U̅, V̅, Ũ, Ṽ = averages

    Hᶠᶜ = x_column_depth(i, j, k_top, grid, filled_halos, ηᵖ)
    Hᶜᶠ = y_column_depth(i, j, k_top, grid, filled_halos, ηᵖ)
    ∂xᵣ = x_derivative_operator(filled_halos)
    ∂yᵣ = y_derivative_operator(filled_halos)
    δx  = x_difference_operator(filled_halos)
    δy  = y_difference_operator(filled_halos)

    # With the frozen forcing this dispatches back to a plain `Gᵁ[i, j, 1]` load.
    Gᵁˢ = slow_forcing(i, j, Gᵁ, Gᵁᶜ, w, sᵐ)
    Gⱽˢ = slow_forcing(i, j, Gⱽ, Gⱽᶜ, w, sᵐ)

    δh_U = (δx(i, j, grid.Nz, grid, Δy_qᶠᶜᶠ, Uᵖ) + δy(i, j, grid.Nz, grid, Δx_qᶜᶠᶠ, Vᵖ)) * Az⁻¹ᶜᶜᶠ(i, j, k_top, grid)

    @inbounds begin
        η[i, j, k_top] = ηⁿ[i, j, k_top] + Δτ * (F(i, j, k_top, grid, clock, (; η = ηᵖ, U = Uᵖ, V = Vᵖ)) - δh_U)
        U[i, j, 1]     = Uⁿ[i, j, 1] + Δτ * (- g * Hᶠᶜ * ∂xᵣ(i, j, k_top, grid, ηᵖ) + Gᵁˢ)
        V[i, j, 1]     = Vⁿ[i, j, 1] + Δτ * (- g * Hᶜᶠ * ∂yᵣ(i, j, k_top, grid, ηᵖ) + Gⱽˢ)

        η̅[i, j, k_top] += averaging_weight * η[i, j, k_top]
        U̅[i, j, 1]     += averaging_weight * U[i, j, 1]
        V̅[i, j, 1]     += averaging_weight * V[i, j, 1]

        # The transport in the tracer-continuity average is the flux that advanced η, i.e. the previous stage.
        Ũ[i, j, 1] += transport_weight * Uᵖ[i, j, 1]
        Ṽ[i, j, 1] += transport_weight * Vᵖ[i, j, 1]
    end
end

# Change name
const FNS = FixedSubstepNumber
const FTS = FixedTimeStepSize

# Since weights can be negative in the first few substeps (as in the default averaging kernel), we set a minimum number
# of substeps to execute to avoid numerical issues.
const MINIMUM_SUBSTEPS = 5

@inline calculate_substeps(substepping::FNS, Δt=nothing) = length(substepping.averaging_weights)
@inline calculate_substeps(substepping::FTS, Δt) = max(MINIMUM_SUBSTEPS, ceil(Int, 2 * Δt / substepping.Δt_barotropic))

@inline calculate_adaptive_settings(substepping::FNS, substeps) = substepping.fractional_step_size, substepping.averaging_weights, substepping.transport_weights
@inline calculate_adaptive_settings(substepping::FTS, substeps) = weights_from_substeps(eltype(substepping.Δt_barotropic), substeps, substepping.averaging_kernel)

#####
##### Halo filling in between barotropic kernels
#####
##### `ExtendedHalos` substeps into the halo rather than filling it, so its argument groups are empty.

@inline fill_barotropic_halos!(free_surface, arch, ::Tuple{}) = nothing

@inline function fill_barotropic_halos!(free_surface, arch, halo_args::Tuple)
    only_local_halos = fill_only_local_halos(free_surface)
    maybe_distributed_fill_halo_regions!(arch, first(halo_args)...; only_local_halos)
    return fill_barotropic_halos!(free_surface, arch, Base.tail(halo_args))
end

# A substep clock with a smaller Δτ is needed for inter-step boundary conditions to be valid.
@inline barotropic_substep_clock(clock, Δτᴮ) =
    (; time = clock.time, iteration = clock.iteration, stage = 0, last_stage_Δt = Δτᴮ)

@inline barotropic_halo_arguments(::SplitExplicitFreeSurface{ExtendedHalos}, arch, grid, substep_clock, reference_fields, fields_to_fill::Tuple) = ()
@inline barotropic_halo_arguments(::FillHaloSplitExplicit, arch, grid, substep_clock, reference_fields, ::Tuple{}) = ()

@inline function barotropic_halo_arguments(free_surface::FillHaloSplitExplicit, arch, grid, substep_clock, reference_fields, fields_to_fill::Tuple)
    @apply_regionally halo_args = build_halo_fill_args(first(fields_to_fill), grid, substep_clock, reference_fields)
    @apply_regionally converted_halo_args = prepare_halo_fill_args(arch, halo_args, grid, free_surface)
    remaining = barotropic_halo_arguments(free_surface, arch, grid, substep_clock, reference_fields, Base.tail(fields_to_fill))
    return (converted_halo_args, remaining...)
end

# `filled_halos` selects the topology-aware operators when the halos go stale in between kernels.
@inline substep_filled_halos(::SplitExplicitFreeSurface{ExtendedHalos}) = Val(false)
@inline substep_filled_halos(::FillHaloSplitExplicit)                   = Val(true)

#####
##### Barotropic substeppers
#####
##### A substepper collects, once per baroclinic step, everything that does not change from substep to substep.
##### We perform ~50 substeps of ~100 very small kernels, whose latency is dominated by argument conversion.

# One method per leading-scalar count: a `weights...` vararg splats through a dynamic call and boxes the
# whole argument list on every launch.
@inline apply_barotropic_kernel!(kernel, args, w₁, w₂)     = kernel(w₁, w₂, args...)
@inline apply_barotropic_kernel!(kernel, args, w₁, w₂, w₃) = kernel(w₁, w₂, w₃, args...)

function barotropic_substepper(timestepper::ForwardBackwardScheme, free_surface, arch, grid, parameters, substep_arguments)

    η       = free_surface.displacement
    U, V    = free_surface.barotropic_velocities
    state   = free_surface.filtered_state
    η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅
    Ũ, Ṽ    = state.Ũ, state.Ṽ

    (; Δτᴮ, clock, F, g, GUⁿ, GVⁿ, Gᵁᶜ, Gⱽᶜ, w) = substep_arguments

    filled_halos = substep_filled_halos(free_surface)

    @apply_regionally velocity_kernel!, _     = configure_kernel(arch, grid, parameters, _split_explicit_barotropic_velocity!)
    @apply_regionally free_surface_kernel!, _ = configure_kernel(arch, grid, parameters, _split_explicit_free_surface!)

    U_args = (grid, filled_halos, Δτᴮ, η, U, V, GUⁿ, GVⁿ, Gᵁᶜ, Gⱽᶜ, w, g, U̅, V̅, timestepper)
    η_args = (grid, filled_halos, Δτᴮ, η, U, V, F, clock, η̅, Ũ, Ṽ, timestepper)

    @apply_regionally converted_U_args = convert_to_device(arch, U_args)
    @apply_regionally converted_η_args = convert_to_device(arch, η_args)

    substep_clock = barotropic_substep_clock(clock, Δτᴮ)
    barotropic_model_fields = (; U, V, η)

    velocity_halos     = barotropic_halo_arguments(free_surface, arch, grid, substep_clock, barotropic_model_fields, (U, V))
    free_surface_halos = barotropic_halo_arguments(free_surface, arch, grid, substep_clock, barotropic_model_fields, (η, ))

    return (; free_surface_kernel!, velocity_kernel!, η_args = converted_η_args, U_args = converted_U_args,
              velocity_halos, free_surface_halos)
end

function barotropic_substepper(timestepper::RungeKutta3Scheme, free_surface, arch, grid, parameters, substep_arguments)

    η       = free_surface.displacement
    U, V    = free_surface.barotropic_velocities
    state   = free_surface.filtered_state
    η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅
    Ũ, Ṽ    = state.Ũ, state.Ṽ

    (; Δτᴮ, clock, F, g, GUⁿ, GVⁿ, Gᵁᶜ, Gⱽᶜ, w) = substep_arguments

    filled_halos = substep_filled_halos(free_surface)

    @apply_regionally stage_kernel!, _ = configure_kernel(arch, grid, parameters, _barotropic_stage!)

    stages_Δτ = stage_parameters(timestepper, Δτᴮ)

    # The buffers rotate in the three stages
    previous_state = ((η, U, V), (timestepper.η¹, timestepper.U¹, timestepper.V¹), (timestepper.η², timestepper.U², timestepper.V²))
    stage_state    = ((timestepper.η¹, timestepper.U¹, timestepper.V¹), (timestepper.η², timestepper.U², timestepper.V²), (η, U, V))

    substep_clock = barotropic_substep_clock(clock, Δτᴮ)

    stage_args = ntuple(Val(length(stages_Δτ))) do stage
        args = (stages_Δτ[stage], grid, filled_halos, (η, U, V), previous_state[stage], stage_state[stage],
                (GUⁿ, GVⁿ, Gᵁᶜ, Gⱽᶜ, w), g, F, clock, (η̅, U̅, V̅, Ũ, Ṽ))

        @apply_regionally converted_args = convert_to_device(arch, args)
        converted_args
    end

    stage_halos = ntuple(Val(length(stages_Δτ))) do stage
        ηᵖ, Uᵖ, Vᵖ = previous_state[stage]
        barotropic_halo_arguments(free_surface, arch, grid, substep_clock, (; U = Uᵖ, V = Vᵖ, η = ηᵖ), (Uᵖ, Vᵖ, ηᵖ))
    end

    return (; stage_kernel!, stage_args, stage_halos)
end

function barotropic_substep!(::ForwardBackwardScheme, substepper, free_surface, arch, averaging_weight, transport_weight, sᵐ)

    fill_barotropic_halos!(free_surface, arch, substepper.free_surface_halos)
    @apply_regionally apply_barotropic_kernel!(substepper.velocity_kernel!, substepper.U_args, averaging_weight, sᵐ)

    fill_barotropic_halos!(free_surface, arch, substepper.velocity_halos)
    @apply_regionally apply_barotropic_kernel!(substepper.free_surface_kernel!, substepper.η_args, averaging_weight, transport_weight)

    return nothing
end

@noinline function barotropic_substep!(::RungeKutta3Scheme, substepper, free_surface, arch, averaging_weight, transport_weight, sᵐ)

    stage_args = substepper.stage_args
    final_stage = lastindex(stage_args)

    @unroll for stage in eachindex(stage_args)
        # Only the stage that ends the substep contributes to the time averages.
        final = stage == final_stage
        aw = ifelse(final, averaging_weight, zero(averaging_weight))
        tw = ifelse(final, transport_weight, zero(transport_weight))

        fill_barotropic_halos!(free_surface, arch, substepper.stage_halos[stage])
        @apply_regionally apply_barotropic_kernel!(substepper.stage_kernel!, stage_args[stage], aw, tw, sᵐ)
    end

    return nothing
end

# `Δt` is the baroclinic step and `Δτᴮ` the barotropic substep.
function iterate_split_explicit!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, Δt, F, clock, weights, transport_weights,
                                 baroclinic_timestepper, ::Val{Nsubsteps}) where Nsubsteps

    # Note: `free_surface.displacement.grid` carries the extended halos, where `grid` may not.
    grid        = free_surface.displacement.grid
    arch        = architecture(grid)
    timestepper = free_surface.timestepper
    parameters  = free_surface.kernel_parameters
    g           = free_surface.gravitational_acceleration

    Gᵁᶜ, Gⱽᶜ, w = stage_reconstruction(free_surface.slow_forcing, baroclinic_timestepper, clock.stage, Δt)

    substep_arguments = (; Δτᴮ, clock, F, g, GUⁿ, GVⁿ, Gᵁᶜ, Gⱽᶜ, w)
    substepper = barotropic_substepper(timestepper, free_surface, arch, grid, parameters, substep_arguments)

    # The substepper holds device values that do not root the fields they point into.
    GC.@preserve free_surface GUⁿ GVⁿ Gᵁᶜ Gⱽᶜ F begin
        @unroll for substep in 1:Nsubsteps
            @inbounds averaging_weight = weights[substep]
            @inbounds transport_weight = transport_weights[substep]
            sᵐ = (substep - oneunit(substep) / 2) * Δτᴮ   # MIDPOINT of the substep, measured from tⁿ

            barotropic_substep!(timestepper, substepper, free_surface, arch, averaging_weight, transport_weight, sᵐ)
        end
    end

    return nothing
end

@kernel function _update_split_explicit_state!(η, U, V, grid, state)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    @inbounds begin
        η[i, j, k_top] = state.η̅[i, j, k_top]
        U[i, j, 1]     = state.U̅[i, j, 1]
        V[i, j, 1]     = state.V̅[i, j, 1]
    end
end

#####
##### SplitExplicitFreeSurface barotropic subcycling
#####

# Open boundaries read model fields while filling the barotropic halos; `ExtendedHalos` has none, so it
# fills without threading them, which avoids a per-step allocation on distributed grids.
@inline fill_barotropic_state_halos!(field, ::SplitExplicitFreeSurface{ExtendedHalos}, model) =
    fill_halo_regions!(field; async=true)
@inline fill_barotropic_state_halos!(field, ::FillHaloSplitExplicit, model) =
    fill_halo_regions!(field, model.clock, fields(model); async=true)

function step_free_surface!(free_surface::SplitExplicitFreeSurface, model, baroclinic_timestepper, Δt)
    # Note: free_surface.displacement.grid != model.grid for DistributedSplitExplicitFreeSurface since
    # halo_size(free_surface.displacement.grid) != halo_size(model.grid)
    free_surface_grid = free_surface.displacement.grid
    filtered_state    = free_surface.filtered_state
    substepping       = free_surface.substepping

    barotropic_velocities = free_surface.barotropic_velocities

    barotropic_timestepper = free_surface.timestepper
    baroclinic_timestepper = model.timestepper

    # Compute barotropic substepping parameters: number of substeps per baroclinic time step, fractional barotropic time
    # step, and the corresponding averaging and transport weights.
    Nsubsteps = calculate_substeps(substepping, Δt)
    fractional_Δt, weights, transport_weights = calculate_adaptive_settings(substepping, Nsubsteps)
    Nsubsteps = length(weights)

    # Barotropic time step in seconds
    Δτᴮ = fractional_Δt * Δt

    # Slow forcing terms
    GUⁿ = model.timestepper.Gⁿ.U
    GVⁿ = model.timestepper.Gⁿ.V

    # Free surface state
    η = free_surface.displacement
    U = barotropic_velocities.U
    V = barotropic_velocities.V
    F = model.forcing.η

    # Wait for setup step to finish.
    wait_free_surface_communication!(free_surface, model, architecture(free_surface_grid))

    # Reset the filtered fields and the barotropic timestepper to zero.
    @apply_regionally initialize_free_surface_state!(free_surface, baroclinic_timestepper, barotropic_timestepper)

    # Solve for the free surface at tⁿ⁺¹.
    iterate_split_explicit!(free_surface, free_surface_grid, GUⁿ, GVⁿ, Δτᴮ, Δt, F, model.clock, weights,
                            transport_weights, baroclinic_timestepper, Val(Nsubsteps))

    # Update eta and velocities for the next timestep. The halos are updated in the `update_state!` function.
    @apply_regionally launch!(architecture(free_surface_grid), free_surface_grid, :xy, _update_split_explicit_state!, η, U, V, free_surface_grid, filtered_state)

    # Fill all the barotropic state.
    fill_barotropic_state_halos!((filtered_state.Ũ, filtered_state.Ṽ), free_surface, model)
    fill_barotropic_state_halos!((U, V), free_surface, model)
    fill_barotropic_state_halos!(η, free_surface, model)

    return nothing
end
