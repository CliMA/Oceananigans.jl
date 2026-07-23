using KernelAbstractions.Extras.LoopInfo: @unroll

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
@inline x_difference_operator(::Val{true})  = δxᶜᵃᵃ
@inline y_difference_operator(::Val{false}) = δyTᵃᶜᵃ
@inline y_difference_operator(::Val{true})  = δyᵃᶜᵃ

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
@kernel function _split_explicit_barotropic_velocity!(averaging_weight, grid, filled_halos, Δτ, η, U, V, Gᵁ, Gⱽ, g, U̅, V̅, timestepper)
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
    @inbounds begin
        U[i, j, 1] += Δτ * (- g * Hᶠᶜ * ∂xᵣ(i, j, k_top, grid, η★, timestepper, η) + Gᵁ[i, j, 1])
        V[i, j, 1] += Δτ * (- g * Hᶜᶠ * ∂yᵣ(i, j, k_top, grid, η★, timestepper, η) + Gⱽ[i, j, 1])

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
##### Multi-stage substep kernels (RungeKutta2Scheme / RungeKutta3Scheme)
#####
##### Each barotropic substep runs three {η-update, U-update} stage pairs from the substep-start state
##### (η⁰, U⁰, V⁰). The averaged quantities (η̅, U̅, V̅, Ũ, Ṽ) are accumulated ONLY on the final stage, from the
##### free surface that ends the substep and the flux U that advances η in that final stage — this is the same
##### continuity-consistent transport as the forward-backward path, so tracer constancy is preserved.
#####
##### Halos are filled between every stage, so the non-topology-aware operators are used throughout.

@kernel function _barotropic_velocity_stage!(averaging_weight, grid, Δτ, ηᵖ, U, V, U⁰, V⁰, Gᵁ, Gⱽ, g, U̅, V̅)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz + 1

    Hᶠᶜ = column_depthᶠᶜᵃ(i, j, k_top, grid, ηᵖ)
    Hᶜᶠ = column_depthᶜᶠᵃ(i, j, k_top, grid, ηᵖ)

    @inbounds begin
        U[i, j, 1] = U⁰[i, j, 1] + Δτ * (- g * Hᶠᶜ * ∂xᵣᶠᶜᶠ(i, j, k_top, grid, ηᵖ) + Gᵁ[i, j, 1])
        V[i, j, 1] = V⁰[i, j, 1] + Δτ * (- g * Hᶜᶠ * ∂yᵣᶜᶠᶠ(i, j, k_top, grid, ηᵖ) + Gⱽ[i, j, 1])

        U̅[i, j, 1] += averaging_weight * U[i, j, 1]
        V̅[i, j, 1] += averaging_weight * V[i, j, 1]
    end
end

@kernel function _barotropic_free_surface_stage!(averaging_weight, transport_weight, grid, Δτ,
                                                 η, η⁰, U, V, F, clock, η̅, Ũ, Ṽ)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz + 1

    δh_U = (δxᶜᵃᵃ(i, j, grid.Nz, grid, Δy_qᶠᶜᶠ, U) +
            δyᵃᶜᵃ(i, j, grid.Nz, grid, Δx_qᶜᶠᶠ, V)) * Az⁻¹ᶜᶜᶠ(i, j, k_top, grid)

    @inbounds begin
        η[i, j, k_top] = η⁰[i, j, k_top] + Δτ * (F(i, j, k_top, grid, clock, (; η, U, V)) - δh_U)
        η̅[i, j, k_top] += averaging_weight * η[i, j, k_top]
        Ũ[i, j, 1]     += transport_weight * U[i, j, 1]
        Ṽ[i, j, 1]     += transport_weight * V[i, j, 1]
    end
end

function iterate_split_explicit_multistage!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, ::Val{Nsubsteps}) where Nsubsteps
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

    η⁰, U⁰, V⁰, ηᵖ = timestepper.η⁰, timestepper.U⁰, timestepper.V⁰, timestepper.ηᵖ

    velocity_kernel!, _     = configure_kernel(arch, grid, parameters, _barotropic_velocity_stage!)
    free_surface_kernel!, _ = configure_kernel(arch, grid, parameters, _barotropic_free_surface_stage!)

    stages = stage_parameters(timestepper, Δτᴮ)

    for substep in 1:Nsubsteps
        @inbounds averaging_weight = weights[substep]
        @inbounds transport_weight = transport_weights[substep]

        # Cache the substep-start state (parent copies to keep halos consistent).
        parent(η⁰) .= parent(η)
        parent(U⁰) .= parent(U)
        parent(V⁰) .= parent(V)

        @unroll for stage in eachindex(stages)
            γ = stages[stage]
            final = stage == lastindex(stages)
            aw = ifelse(final, averaging_weight, zero(averaging_weight))
            tw = ifelse(final, transport_weight, zero(transport_weight))

            # Save the previous-stage free surface, then advance η (from η⁰), then U (using the saved ηᵖ).
            parent(ηᵖ) .= parent(η)

            free_surface_kernel!(aw, tw, grid, γ, η, η⁰, U, V, F, clock, η̅, Ũ, Ṽ)
            fill_halo_regions!(η)

            velocity_kernel!(aw, grid, γ, ηᵖ, U, V, U⁰, V⁰, GUⁿ, GVⁿ, g, U̅, V̅)
            fill_halo_regions!((U, V))
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

iterate_split_explicit!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, ::Val{Nsubsteps}) where Nsubsteps =
    @apply_regionally iterate_split_explicit_in_halo!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, Val(Nsubsteps))

function iterate_split_explicit!(free_surface::FillHaloSplitExplicit, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, ::Val{Nsubsteps}) where Nsubsteps
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
        iterate_split_explicit_multistage!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, Val(Nsubsteps))
        return nothing
    end

    @apply_regionally velocity_kernel!, _     = configure_kernel(arch, grid, parameters, _split_explicit_barotropic_velocity!)
    @apply_regionally free_surface_kernel!, _ = configure_kernel(arch, grid, parameters, _split_explicit_free_surface!)

    U_args = (grid, Val(true), Δτᴮ, η, U, V, GUⁿ, GVⁿ, g, U̅, V̅, timestepper)
    η_args = (grid, Val(true), Δτᴮ, η, U, V, F, clock, η̅, Ũ, Ṽ, timestepper)

    GC.@preserve U_args η_args begin
        # We need to perform ~50 time-steps which means launching ~100 very small kernels: we are limited by latency of
        # argument conversion to GPU-compatible values. To alleviate this penalty we convert first and then we substep!
        @apply_regionally converted_U_args = convert_to_device(arch, U_args)
        @apply_regionally converted_η_args = convert_to_device(arch, η_args)

        @unroll for substep in 1:Nsubsteps
            @inbounds averaging_weight = weights[substep]
            @inbounds transport_weight = transport_weights[substep]

            fill_halo_regions!((U, V))
            @apply_regionally apply_barotropic_kernel!(free_surface_kernel!, averaging_weight, transport_weight, converted_η_args)

            fill_halo_regions!(η)
            @apply_regionally apply_barotropic_kernel!(velocity_kernel!, averaging_weight, converted_U_args)
        end
    end

    return nothing
end

@inline apply_barotropic_kernel!(kernel, weight, args) = kernel(weight, args...)
@inline apply_barotropic_kernel!(kernel, w1, w2, args) = kernel(w1, w2, args...)

function iterate_split_explicit_in_halo!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, ::Val{Nsubsteps}) where Nsubsteps
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
        iterate_split_explicit_multistage!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, Val(Nsubsteps))
        return nothing
    end

    barotropic_velocity_kernel!, _ = configure_kernel(arch, grid, parameters, _split_explicit_barotropic_velocity!)
    free_surface_kernel!, _        = configure_kernel(arch, grid, parameters, _split_explicit_free_surface!)

    U_args = (grid, Val(false), Δτᴮ, η, U, V, GUⁿ, GVⁿ, g, U̅, V̅, timestepper)
    η_args = (grid, Val(false), Δτᴮ, η, U, V, F, clock, η̅, Ũ, Ṽ, timestepper)

    GC.@preserve U_args η_args begin
        # We need to perform ~50 time-steps which means launching ~100 very small kernels: we are limited by latency of
        # argument conversion to GPU-compatible values. To alleviate this penalty we convert first and then we substep!
        converted_U_args = convert_to_device(arch, U_args)
        converted_η_args = convert_to_device(arch, η_args)

        @unroll for substep in 1:Nsubsteps
            @inbounds averaging_weight = weights[substep]
            @inbounds transport_weight = transport_weights[substep]

            free_surface_kernel!(averaging_weight, transport_weight, converted_η_args...)
            barotropic_velocity_kernel!(averaging_weight, converted_U_args...)
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
    iterate_split_explicit!(free_surface, free_surface_grid, GUⁿ, GVⁿ, Δτᴮ, F, model.clock, weights, transport_weights, Val(Nsubsteps))

    # Update eta and velocities for the next timestep. The halos are updated in the `update_state!` function.
    @apply_regionally launch!(architecture(free_surface_grid), free_surface_grid, :xy, _update_split_explicit_state!, η, U, V, free_surface_grid, filtered_state)

    # Fill all the barotropic state.
    fill_halo_regions!((filtered_state.Ũ, filtered_state.Ṽ); async=true)
    fill_halo_regions!((U, V); async=true)
    fill_halo_regions!(η; async=true)

    return nothing
end
