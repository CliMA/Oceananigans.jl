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

    # ∂τ(U) = - ∇η★ + G, using the free surface η★ that already includes the just-updated ηᵐ⁺¹ (the backward
    # half of the forward-backward step). Note: use ∂xᵣT/∂yᵣT (derivatives at constant r), since η lives on the
    # surface and has no vertical structure.
    # With the frozen forcing this dispatches back to a plain `Gᵁ[i, j, 1]` load.
    Gᵁˢ = slow_forcing(i, j, Gᵁ, Gᵁᶜ, w, sᵐ)
    Gⱽˢ = slow_forcing(i, j, Gⱽ, Gⱽᶜ, w, sᵐ)

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

        # Time-averaging the free surface, and the transport U★/V★ that advanced it (constancy); for plain
        # forward-backward U★/V★ is simply the current velocity.
        η̅[i, j, k_top] += averaging_weight * η[i, j, k_top]
        Ũ[i, j, 1]     += transport_weight * U★(i, j, 1, grid, timestepper, U)
        Ṽ[i, j, 1]     += transport_weight * V★(i, j, 1, grid, timestepper, V)
    end
end

#####
##### Multi-stage substep kernel (RungeKutta3Scheme)
#####
##### Each barotropic substep runs three RK stages from the substep-start state (η⁰, U⁰, V⁰). Within a stage the
##### free surface and the velocity are advanced together in one kernel: they depend only on the previous-stage
##### state, the free surface through the transport (ηᵖ, Uᵖ, Vᵖ) and the velocity through the free surface, so
##### they commute. Fusing them requires the previous-stage copies, since a kernel that read the live `U` while
##### another thread wrote it would race on the `δx` stencil.
#####
##### The averaged quantities (η̅, U̅, V̅, Ũ, Ṽ) are accumulated ONLY on the final stage, from the free surface
##### that ends the substep and the flux U that advances η in that stage — the same continuity-consistent
##### transport as the forward-backward path, so tracer constancy is preserved.
#####
##### Halos are filled once per stage, so the non-topology-aware operators are used throughout.


@kernel function _barotropic_stage!(averaging_weight, transport_weight, grid, Δτ,
                                    η, η⁰, ηᵖ,
                                    U, V, U⁰, V⁰, Uᵖ, Vᵖ,
                                    Gᵁ, Gⱽ, Gᵁᶜ, Gⱽᶜ,
                                    w, s, g, F, clock,
                                    η̅, U̅, V̅, Ũ, Ṽ)

    i, j = @index(Global, NTuple)
    k_top = grid.Nz + 1

    Hᶠᶜ = column_depthᶠᶜᵃ(i, j, k_top, grid, ηᵖ)
    Hᶜᶠ = column_depthᶜᶠᵃ(i, j, k_top, grid, ηᵖ)

    # With the frozen forcing this dispatches back to a plain `Gᵁ[i, j, 1]` load.
    Gᵁˢ = slow_forcing(i, j, Gᵁ, Gᵁᶜ, w, s)
    Gⱽˢ = slow_forcing(i, j, Gⱽ, Gⱽᶜ, w, s)

    δh_U = (δxᶜᶜᶜ(i, j, grid.Nz, grid, Δy_qᶠᶜᶠ, Uᵖ) + δyᶜᶜᶜ(i, j, grid.Nz, grid, Δx_qᶜᶠᶠ, Vᵖ)) * Az⁻¹ᶜᶜᶠ(i, j, k_top, grid)

    @inbounds begin
        η[i, j, k_top] = η⁰[i, j, k_top] + Δτ * (F(i, j, k_top, grid, clock, (; η = ηᵖ, U = Uᵖ, V = Vᵖ)) - δh_U)
        η̅[i, j, k_top] += averaging_weight * η[i, j, k_top]
        # The transport carried into the tracer-continuity average is the flux that advanced η above, which
        # is the previous-stage velocity -- read from `Uᵖ` rather than from `U`, which is overwritten below.
        Ũ[i, j, 1]     += transport_weight * Uᵖ[i, j, 1]
        Ṽ[i, j, 1]     += transport_weight * Vᵖ[i, j, 1]

        U[i, j, 1] = U⁰[i, j, 1] + Δτ * (- g * Hᶠᶜ * ∂xᵣᶠᶜᶠ(i, j, k_top, grid, ηᵖ) + Gᵁˢ)
        V[i, j, 1] = V⁰[i, j, 1] + Δτ * (- g * Hᶜᶠ * ∂yᵣᶜᶠᶠ(i, j, k_top, grid, ηᵖ) + Gⱽˢ)
        U̅[i, j, 1] += averaging_weight * U[i, j, 1]
        V̅[i, j, 1] += averaging_weight * V[i, j, 1]
    end
end

function iterate_split_explicit_multistage!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, Δt, F, clock, weights, transport_weights, baroclinic_timestepper, ::Val{Nsubsteps}) where Nsubsteps
    arch        = architecture(grid)
    η           = free_surface.displacement
    grid        = free_surface.displacement.grid
    state       = free_surface.filtered_state
    timestepper = free_surface.timestepper
    g           = free_surface.gravitational_acceleration
    parameters  = free_surface.kernel_parameters

    U, V    = free_surface.barotropic_velocities
    η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅
    Ũ, Ṽ    = state.Ũ, state.Ṽ

    η⁰, U⁰, V⁰, ηᵖ, Uᵖ, Vᵖ = timestepper.η⁰, timestepper.U⁰, timestepper.V⁰, timestepper.ηᵖ, timestepper.Uᵖ, timestepper.Vᵖ

    barotropic_stage_kernel!, _ = configure_kernel(arch, grid, parameters, _barotropic_stage!)

    stages_Δτ = stage_parameters(timestepper, Δτᴮ)

    # `nothing` caches wherever the stage's polynomial is a constant, which dispatches the substep kernel back
    # to the plain frozen load.
    Gᵁᶜ, Gⱽᶜ, w = stage_reconstruction(free_surface.slow_forcing, baroclinic_timestepper, clock.stage, Δt)

    for substep in 1:Nsubsteps
        @inbounds averaging_weight = weights[substep]
        @inbounds transport_weight = transport_weights[substep]

        # MIDPOINT of the substep, measured from tⁿ. Sampling at either endpoint costs two orders.
        s = (substep - oneunit(substep) / 2) * Δτᴮ

        # Cache the substep-start state (parent copies to keep halos consistent).
        parent(η⁰) .= parent(η)
        parent(U⁰) .= parent(U)
        parent(V⁰) .= parent(V)

        @unroll for stage in eachindex(stages_Δτ)
            Δτ = stages_Δτ[stage]
            final = stage == lastindex(stages_Δτ)
            aw = ifelse(final, averaging_weight, zero(averaging_weight))
            tw = ifelse(final, transport_weight, zero(transport_weight))

            # Save the previous-stage free surface
            parent(ηᵖ) .= parent(η)
            parent(Uᵖ) .= parent(U)
            parent(Vᵖ) .= parent(V)

            barotropic_stage_kernel!(aw, tw, grid, Δτ, η, η⁰, ηᵖ, U, V, U⁰, V⁰, Uᵖ, Vᵖ, GUⁿ, GVⁿ, Gᵁᶜ, Gⱽᶜ, w, s, g, F, clock, η̅, U̅, V̅, Ũ, Ṽ)
            fill_halo_regions!((η, U, V))
        end
    end

    return nothing
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

# `Δt` is the baroclinic step and `Δτᴮ` the barotropic substep; both are needed, since a slow-forcing
# reconstruction is a polynomial in time across the whole baroclinic step. `baroclinic_timestepper` supplies the
# stage nodes that polynomial is built on, and is `nothing` when there is no reconstruction to build.
function iterate_split_explicit!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, Δt, F, clock, weights, transport_weights,
                                 baroclinic_timestepper, ::Val{Nsubsteps}) where Nsubsteps
    @apply_regionally iterate_split_explicit_in_halo!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, Δt, F, clock, weights,
                                                     transport_weights, baroclinic_timestepper, Val(Nsubsteps))
    return nothing
end

function iterate_split_explicit!(free_surface::FillHaloSplitExplicit, grid, GUⁿ, GVⁿ, Δτᴮ, Δt, F, clock, weights,
                                 transport_weights, baroclinic_timestepper, ::Val{Nsubsteps}) where Nsubsteps

    η           = free_surface.displacement
    grid        = free_surface.displacement.grid
    arch        = architecture(grid)
    state       = free_surface.filtered_state
    timestepper = free_surface.timestepper
    g           = free_surface.gravitational_acceleration
    parameters  = free_surface.kernel_parameters

    # Unpack state quantities, parameters and forcing terms.
    U, V    = free_surface.barotropic_velocities
    η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅
    Ũ, Ṽ    = state.Ũ, state.Ṽ

    if requires_multistage(timestepper)
        iterate_split_explicit_multistage!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, Δt, F, clock, weights, transport_weights,
                                          baroclinic_timestepper, Val(Nsubsteps))
        return nothing
    end

    @apply_regionally velocity_kernel!, _     = configure_kernel(arch, grid, parameters, _split_explicit_barotropic_velocity!)
    @apply_regionally free_surface_kernel!, _ = configure_kernel(arch, grid, parameters, _split_explicit_free_surface!)

    Gᵁᶜ, Gⱽᶜ, w = stage_reconstruction(free_surface.slow_forcing, baroclinic_timestepper, clock.stage, Δt)

    U_args = (grid, Val(true), Δτᴮ, η, U, V, GUⁿ, GVⁿ, Gᵁᶜ, Gⱽᶜ, w, g, U̅, V̅, timestepper)
    η_args = (grid, Val(true), Δτᴮ, η, U, V, F, clock, η̅, Ũ, Ṽ, timestepper)

    barotropic_model_fields = (; U, V, η)

    # a substep clock with a smaller Δτ is needed for inter-step boundary conditions to be valid
    substep_clock = (; time = clock.time, iteration = clock.iteration, stage = 0, last_stage_Δt = Δτᴮ)
    @apply_regionally U_halo_args = build_halo_fill_args(U, grid, substep_clock, barotropic_model_fields)
    @apply_regionally V_halo_args = build_halo_fill_args(V, grid, substep_clock, barotropic_model_fields)
    @apply_regionally η_halo_args = build_halo_fill_args(η, grid, substep_clock, barotropic_model_fields)

    only_local_halos = fill_only_local_halos(free_surface)

    GC.@preserve U_args η_args U_halo_args V_halo_args η_halo_args begin
        # We need to perform ~50 time-steps which means launching ~100 very small kernels: we are limited by latency of
        # argument conversion to GPU-compatible values. To alleviate this penalty we convert first and then we substep!
        @apply_regionally converted_U_args = convert_to_device(arch, U_args)
        @apply_regionally converted_η_args = convert_to_device(arch, η_args)
        @apply_regionally converted_U_halo_args = prepare_halo_fill_args(arch, U_halo_args, grid, free_surface)
        @apply_regionally converted_V_halo_args = prepare_halo_fill_args(arch, V_halo_args, grid, free_surface)
        @apply_regionally converted_η_halo_args = prepare_halo_fill_args(arch, η_halo_args, grid, free_surface)

        @unroll for substep in 1:Nsubsteps
            @inbounds averaging_weight = weights[substep]
            @inbounds transport_weight = transport_weights[substep]
            sᵐ = (substep - oneunit(substep) / 2) * Δτᴮ   # MIDPOINT of the substep, measured from tⁿ

            maybe_distributed_fill_halo_regions!(arch, converted_U_halo_args...; only_local_halos)
            maybe_distributed_fill_halo_regions!(arch, converted_V_halo_args...; only_local_halos)
            @apply_regionally apply_barotropic_kernel!(free_surface_kernel!, averaging_weight, transport_weight, converted_η_args)

            maybe_distributed_fill_halo_regions!(arch, converted_η_halo_args...; only_local_halos)
            @apply_regionally apply_barotropic_kernel!(velocity_kernel!, averaging_weight, sᵐ, converted_U_args)
        end
    end

    return nothing
end

@inline apply_barotropic_kernel!(kernel, weight, args) = kernel(weight, args...)
@inline apply_barotropic_kernel!(kernel, w1, w2, args) = kernel(w1, w2, args...)

function iterate_split_explicit_in_halo!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, Δt, F, clock, weights, transport_weights,
                                        baroclinic_timestepper, ::Val{Nsubsteps}) where Nsubsteps
    arch = architecture(grid)

    η           = free_surface.displacement
    grid        = free_surface.displacement.grid
    state       = free_surface.filtered_state
    timestepper = free_surface.timestepper
    g           = free_surface.gravitational_acceleration
    parameters  = free_surface.kernel_parameters

    # Unpack state quantities, parameters and forcing terms.
    U, V    = free_surface.barotropic_velocities
    η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅
    Ũ, Ṽ    = state.Ũ, state.Ṽ

    if requires_multistage(timestepper)
        iterate_split_explicit_multistage!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, Δt, F, clock, weights, transport_weights,
                                          baroclinic_timestepper, Val(Nsubsteps))
        return nothing
    end

    barotropic_velocity_kernel!, _ = configure_kernel(arch, grid, parameters, _split_explicit_barotropic_velocity!)
    free_surface_kernel!, _        = configure_kernel(arch, grid, parameters, _split_explicit_free_surface!)

    Gᵁᶜ, Gⱽᶜ, w = stage_reconstruction(free_surface.slow_forcing, baroclinic_timestepper, clock.stage, Δt)

    U_args = (grid, Val(false), Δτᴮ, η, U, V, GUⁿ, GVⁿ, Gᵁᶜ, Gⱽᶜ, w, g, U̅, V̅, timestepper)
    η_args = (grid, Val(false), Δτᴮ, η, U, V, F, clock, η̅, Ũ, Ṽ, timestepper)

    GC.@preserve U_args η_args begin
        # We need to perform ~50 time-steps which means launching ~100 very small kernels: we are limited by latency of
        # argument conversion to GPU-compatible values. To alleviate this penalty we convert first and then we substep!
        converted_U_args = convert_to_device(arch, U_args)
        converted_η_args = convert_to_device(arch, η_args)

        @unroll for substep in 1:Nsubsteps
            @inbounds averaging_weight = weights[substep]
            @inbounds transport_weight = transport_weights[substep]
            sᵐ = (substep - oneunit(substep) / 2) * Δτᴮ   # MIDPOINT of the substep, measured from tⁿ

            free_surface_kernel!(averaging_weight, transport_weight, converted_η_args...)
            barotropic_velocity_kernel!(averaging_weight, sᵐ, converted_U_args...)
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
