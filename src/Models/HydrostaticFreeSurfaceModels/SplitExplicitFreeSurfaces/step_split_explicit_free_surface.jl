using Oceananigans: fields
using Oceananigans.DistributedComputations: maybe_distributed_fill_halo_regions!
using Oceananigans.BoundaryConditions: BoundaryCondition, NormalFlow, GravityWaveRadiation, has_target_transport, get_target_transport
using KernelAbstractions.Extras.LoopInfo: @unroll

#####
##### Targeted barotropic transports through GravityWaveRadiation boundaries
#####

const FlatherBC = BoundaryCondition{<:NormalFlow{<:GravityWaveRadiation}}

# Multi-region fields carry a `MultiRegionObject` of conditions, which cannot hold a target.
side_condition(bcs::FieldBoundaryConditions, side) = getproperty(bcs, side)
side_condition(bcs, side) = nothing

targeted_side(bc) = false
targeted_side(bc::FlatherBC) = has_target_transport(bc.classification.scheme)

has_targeted_barotropic_sides(U_bcs, V_bcs) =
    targeted_side(side_condition(U_bcs, :west))  || targeted_side(side_condition(U_bcs, :east)) ||
    targeted_side(side_condition(V_bcs, :south)) || targeted_side(side_condition(V_bcs, :north))

# Checked on the conditions the user passed to the model, before they are split into ranks or regions,
# so every rank reaches the same verdict. The face integrals below are rank-local, hence the restriction;
# multi-region grids add their own method.
function validate_free_surface_boundary_conditions(::SplitExplicitFreeSurface, boundary_conditions, grid)
    targeted = has_targeted_barotropic_sides(get(boundary_conditions, :U, nothing), get(boundary_conditions, :V, nothing))
    if targeted && grid isa DistributedGrid
        throw(ArgumentError("`target_transport` on `GravityWaveRadiation` boundary conditions is not supported on distributed grids."))
    end
    return nothing
end

# Every targeted side stores its target, the wet length of its face and a reduced `Field` holding the face
# integral of the transport, computed on the device each substep like the fields in `Models.boundary_transport`.
# One group of sides pins `U` and `V` within the substeps, the other the filtered `Ũ` and `Ṽ` afterwards.
function materialize_barotropic_boundary_transport(U, V, Ũ, Ṽ, grid)
    has_targeted_barotropic_sides(U.boundary_conditions, V.boundary_conditions) || return nothing
    return (; barotropic = side_transports(U, V, grid), filtered = side_transports(Ũ, Ṽ, grid))
end

barotropic_sides(::Nothing) = nothing
barotropic_sides(boundary_transport) = boundary_transport.barotropic

filtered_sides(::Nothing) = nothing
filtered_sides(boundary_transport) = boundary_transport.filtered

side_transports(U, V, grid) = (west  = side_transport(side_condition(U.boundary_conditions, :west),  U, grid, Val(:west)),
                               east  = side_transport(side_condition(U.boundary_conditions, :east),  U, grid, Val(:east)),
                               south = side_transport(side_condition(V.boundary_conditions, :south), V, grid, Val(:south)),
                               north = side_transport(side_condition(V.boundary_conditions, :north), V, grid, Val(:north)))

side_transport(bc, field, grid, side) = nothing

function side_transport(bc::FlatherBC, field, grid, side)
    has_target_transport(bc.classification.scheme) || return nothing
    wet_length = face_wet_length(field, grid, side)
    wet_length > 0 || return nothing # a fully dry side carries no transport
    target = get_target_transport(bc.classification.scheme, grid)
    return (; target, wet_length, integral = face_integral(field, grid, side))
end

# Reductions over immersed grids skip dry columns, so the integrals only see the wet part of the face
face_integral(U, grid, ::Val{:west})  = Field(Integral(view(U, 1, :, :), dims = 2))
face_integral(U, grid, ::Val{:east})  = Field(Integral(view(U, grid.Nx + 1, :, :), dims = 2))
face_integral(V, grid, ::Val{:south}) = Field(Integral(view(V, :, 1, :), dims = 1))
face_integral(V, grid, ::Val{:north}) = Field(Integral(view(V, :, grid.Ny + 1, :), dims = 1))

function face_wet_length(field, grid, side)
    LX, LY, LZ = location(field)
    ones = Field{LX, LY, LZ}(grid)
    set!(ones, 1)
    return @allowscalar compute!(face_integral(ones, grid, side))[]
end

# The face is shifted uniformly over its wet columns so that its integral matches the target
@kernel function _pin_barotropic_face!(U, grid, i, ∫U, target, wet_length, ::Val{:x})
    j = @index(Global, Linear)
    @inbounds shift = (∫U[i, 1, 1] - target) / wet_length
    wet = column_depthᶠᶜᵃ(i, j, grid) > 0
    @inbounds U[i, j, 1] -= ifelse(wet, shift, zero(shift))
end

@kernel function _pin_barotropic_face!(V, grid, j, ∫V, target, wet_length, ::Val{:y})
    i = @index(Global, Linear)
    @inbounds shift = (∫V[1, j, 1] - target) / wet_length
    wet = column_depthᶜᶠᵃ(i, j, grid) > 0
    @inbounds V[i, j, 1] -= ifelse(wet, shift, zero(shift))
end

face_parameters(grid, ::Val{:x}) = KernelParameters(1:grid.Ny)
face_parameters(grid, ::Val{:y}) = KernelParameters(1:grid.Nx)

pin_barotropic_face!(arch, grid, field, index, ::Nothing, direction) = nothing

function pin_barotropic_face!(arch, grid, field, index, side, direction)
    compute!(side.integral)
    launch!(arch, grid, face_parameters(grid, direction), _pin_barotropic_face!,
            field, grid, index, side.integral, side.target, side.wet_length, direction)
    return nothing
end

enforce_barotropic_transport_targets!(arch, grid, U, V, ::Nothing) = nothing

function enforce_barotropic_transport_targets!(arch, grid, U, V, sides)
    pin_barotropic_face!(arch, grid, U, 1,           sides.west,  Val(:x))
    pin_barotropic_face!(arch, grid, U, grid.Nx + 1, sides.east,  Val(:x))
    pin_barotropic_face!(arch, grid, V, 1,           sides.south, Val(:y))
    pin_barotropic_face!(arch, grid, V, grid.Ny + 1, sides.north, Val(:y))
    return nothing
end

# Re-pin the transports after the end-of-step Flather refills, so the barotropic corrector and any
# diagnostics see the same face transports the substeps used.
function enforce_barotropic_transport_targets!(free_surface, grid)
    arch = architecture(grid)
    U, V = free_surface.barotropic_velocities
    Ũ, Ṽ = free_surface.filtered_state.Ũ, free_surface.filtered_state.Ṽ
    enforce_barotropic_transport_targets!(arch, grid, U, V, barotropic_sides(free_surface.boundary_transport))
    enforce_barotropic_transport_targets!(arch, grid, Ũ, Ṽ, filtered_sides(free_surface.boundary_transport))
    return nothing
end

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
# variables (U, V, Ū, V̄) are barotropic fields (`ReducedField`) for which a k index is not defined.
@kernel function _split_explicit_barotropic_velocity!(transport_weight, grid, filled_halos, Δτ, η, U, V, Gᵁ, Gⱽ, g, Ũ, Ṽ, timestepper)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    cache_previous_velocities!(timestepper, i, j, 1, U, V)

    Hᶠᶜ = x_column_depth(i, j, k_top, grid, filled_halos, η) # topology-aware column
    Hᶜᶠ = y_column_depth(i, j, k_top, grid, filled_halos, η) # topology-aware column
    ∂xᵣ = x_derivative_operator(filled_halos)
    ∂yᵣ = y_derivative_operator(filled_halos)

    # ∂τ(U) = - ∇η + G
    # Note: use ∂xᵣT and ∂yᵣT (derivatives at constant r) for the free surface,
    # since η lives on the surface and doesn't have vertical structure
    @inbounds begin
        U[i, j, 1] += Δτ * (- g * Hᶠᶜ * ∂xᵣ(i, j, k_top, grid, η★, timestepper, η) + Gᵁ[i, j, 1])
        V[i, j, 1] += Δτ * (- g * Hᶜᶠ * ∂yᵣ(i, j, k_top, grid, η★, timestepper, η) + Gⱽ[i, j, 1])

        # Averaging the transport
        Ũ[i, j, 1] += transport_weight * U[i, j, 1]
        Ṽ[i, j, 1] += transport_weight * V[i, j, 1]
    end
end

@kernel function _split_explicit_free_surface!(averaging_weight, grid, filled_halos, Δτ, η, U, V, F, clock, η̅, U̅, V̅, timestepper)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    cache_previous_free_surface!(timestepper, i, j, k_top, η)

    δx = x_difference_operator(filled_halos)
    δy = y_difference_operator(filled_halos)

    δh_U = (δx(i, j, grid.Nz, grid, Δy_qᶠᶜᶠ, U★, timestepper, U) +
            δy(i, j, grid.Nz, grid, Δx_qᶜᶠᶠ, U★, timestepper, V)) * Az⁻¹ᶜᶜᶠ(i, j, k_top, grid)

    @inbounds begin
        η[i, j, k_top] += Δτ * (F(i, j, k_top, grid, clock, (; η, U, V)) - δh_U)

        # Time-averaging
        η̅[i, j, k_top] += averaging_weight * η[i, j, k_top]
        U̅[i, j, 1]     += averaging_weight * U[i, j, 1]
        V̅[i, j, 1]     += averaging_weight * V[i, j, 1]
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

iterate_split_explicit!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, ::Val{Nsubsteps}) where Nsubsteps =
    @apply_regionally iterate_split_explicit_in_halo!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, Val(Nsubsteps))

function iterate_split_explicit!(free_surface::FillHaloSplitExplicit, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, ::Val{Nsubsteps}) where Nsubsteps
    arch = architecture(grid)

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
    Ũ, Ṽ    = state.Ũ, state.Ṽ

    @apply_regionally velocity_kernel!, _     = configure_kernel(arch, grid, parameters, _split_explicit_barotropic_velocity!)
    @apply_regionally free_surface_kernel!, _ = configure_kernel(arch, grid, parameters, _split_explicit_free_surface!)

    U_args = (grid, Val(true), Δτᴮ, η, U, V, GUⁿ, GVⁿ, g, Ũ, Ṽ, timestepper)
    η_args = (grid, Val(true), Δτᴮ, η, U, V, F, clock, η̅, U̅, V̅, timestepper)

    barotropic_model_fields = (; U, V, η)

    # a substep clock with a smaller Δτ is needed for inter-step boundary conditions to be valid
    substep_clock = (; time = clock.time, iteration = clock.iteration, stage = 0, last_stage_Δt = Δτᴮ)
    @apply_regionally U_halo_args = build_halo_fill_args(U, grid, substep_clock, barotropic_model_fields)
    @apply_regionally V_halo_args = build_halo_fill_args(V, grid, substep_clock, barotropic_model_fields)
    @apply_regionally η_halo_args = build_halo_fill_args(η, grid, substep_clock, barotropic_model_fields)

    only_local_halos = fill_only_local_halos(free_surface)

    boundary_transport = barotropic_sides(free_surface.boundary_transport)

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

            maybe_distributed_fill_halo_regions!(arch, converted_η_halo_args...; only_local_halos)
            @apply_regionally apply_barotropic_kernel!(velocity_kernel!, transport_weight, converted_U_args)

            maybe_distributed_fill_halo_regions!(arch, converted_U_halo_args...; only_local_halos)
            maybe_distributed_fill_halo_regions!(arch, converted_V_halo_args...; only_local_halos)
            enforce_barotropic_transport_targets!(arch, grid, U, V, boundary_transport)
            @apply_regionally apply_barotropic_kernel!(free_surface_kernel!, averaging_weight, converted_η_args)
        end
    end

    return nothing
end

@inline apply_barotropic_kernel!(kernel, weight, args) = kernel(weight, args...)

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
    Ũ, Ṽ    = state.Ũ, state.Ṽ

    barotropic_velocity_kernel!, _ = configure_kernel(arch, grid, parameters, _split_explicit_barotropic_velocity!)
    free_surface_kernel!, _        = configure_kernel(arch, grid, parameters, _split_explicit_free_surface!)

    U_args = (grid, Val(false), Δτᴮ, η, U, V, GUⁿ, GVⁿ, g, Ũ, Ṽ, timestepper)
    η_args = (grid, Val(false), Δτᴮ, η, U, V, F, clock, η̅, U̅, V̅, timestepper)

    GC.@preserve U_args η_args begin
        # We need to perform ~50 time-steps which means launching ~100 very small kernels: we are limited by latency of
        # argument conversion to GPU-compatible values. To alleviate this penalty we convert first and then we substep!
        converted_U_args = convert_to_device(arch, U_args)
        converted_η_args = convert_to_device(arch, η_args)

        @unroll for substep in 1:Nsubsteps
            @inbounds averaging_weight = weights[substep]
            @inbounds transport_weight = transport_weights[substep]

            barotropic_velocity_kernel!(transport_weight, converted_U_args...)
            free_surface_kernel!(averaging_weight, converted_η_args...)
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
    iterate_split_explicit!(free_surface, free_surface_grid, GUⁿ, GVⁿ, Δτᴮ, F, model.clock, weights, transport_weights, Val(Nsubsteps))

    # Update eta and velocities for the next timestep. The halos are updated in the `update_state!` function.
    @apply_regionally launch!(architecture(free_surface_grid), free_surface_grid, :xy, _update_split_explicit_state!, η, U, V, free_surface_grid, filtered_state)

    # Fill all the barotropic state.
    fill_barotropic_state_halos!((filtered_state.Ũ, filtered_state.Ṽ), free_surface, model)
    fill_barotropic_state_halos!((U, V), free_surface, model)
    fill_barotropic_state_halos!(η, free_surface, model)
    enforce_barotropic_transport_targets!(free_surface, free_surface_grid)

    return nothing
end
