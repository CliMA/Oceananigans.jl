using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

#####
##### Barotropic Coriolis inside the substeps (`CoriolisForwardBackwardScheme`)
#####

const CoriolisSplitExplicit = SplitExplicitFreeSurface{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:CoriolisForwardBackwardScheme}

const FillHaloCoriolisSplitExplicit = SplitExplicitFreeSurface{<:Union{LocalHaloFilling, CompleteHaloFilling},
                                                               <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                               <:CoriolisForwardBackwardScheme}

substep_coriolis_args(timestepper, model) = ()
substep_coriolis_args(::CoriolisForwardBackwardScheme, model) = (model.coriolis,)

#####
##### Reference transports: the barotropic transports whose Coriolis force is already in `Gᵁ` and `Gⱽ`
#####

update_reference_transports!(timestepper, baroclinic_timestepper, U, V) = nothing

function update_reference_transports!(timestepper::CoriolisForwardBackwardScheme, baroclinic_timestepper::QuasiAdamsBashforth2TimeStepper, U, V)
    FT = eltype(U)
    α = convert(FT, 3/2) + baroclinic_timestepper.χ
    β = convert(FT, 1/2) + baroclinic_timestepper.χ

    Uʳ, Vʳ = timestepper.reference_transports
    U⁻, V⁻ = timestepper.previous_transports

    parent(Uʳ) .= α .* parent(U) .- β .* parent(U⁻)
    parent(Vʳ) .= α .* parent(V) .- β .* parent(V⁻)
    parent(U⁻) .= parent(U)
    parent(V⁻) .= parent(V)

    return nothing
end

function update_reference_transports!(timestepper::CoriolisForwardBackwardScheme, ::SplitRungeKuttaTimeStepper, U, V)
    parent(timestepper.reference_transports.U) .= parent(U)
    parent(timestepper.reference_transports.V) .= parent(V)
    return nothing
end

#####
##### Barotropic Coriolis operators (enstrophy-conserving form, as `EnstrophyConserving` in 3D)
#####

@inline horizontal_peripheral_node(i, j, grid, ℓx, ℓy) = peripheral_node(i, j, 1, grid, ℓx, ℓy, Center())
@inline horizontal_peripheral_node(i, j, grid::ImmersedBoundaryGrid, ℓx, ℓy) = horizontal_peripheral_node(i, j, grid.underlying_grid, ℓx, ℓy)

@inline Δx_δVᶜᶠᶠ(i, j, k, grid, V, Vʳ) = @inbounds Δxᶜᶠᶠ(i, j, k, grid) * (V[i, j, 1] - Vʳ[i, j, 1])
@inline Δy_δUᶠᶜᶠ(i, j, k, grid, U, Uʳ) = @inbounds Δyᶠᶜᶠ(i, j, k, grid) * (U[i, j, 1] - Uʳ[i, j, 1])

@inline function barotropic_x_coriolis(i, j, k, grid, coriolis, V, Vʳ, H)
    f̄  = ℑyᵃᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis)
    fV = f̄ * ℑxyᶠᶜᵃ(i, j, k, grid, Δx_δVᶜᶠᶠ, V, Vʳ) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
    active = !horizontal_peripheral_node(i, j, grid, Face(), Center()) & (H > 0)
    return ifelse(active, fV, zero(grid))
end

@inline function barotropic_y_coriolis(i, j, k, grid, coriolis, U, Uʳ, H)
    f̄  = ℑxᶜᵃᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis)
    fU = f̄ * ℑxyᶜᶠᵃ(i, j, k, grid, Δy_δUᶠᶜᶠ, U, Uʳ) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)
    active = !horizontal_peripheral_node(i, j, grid, Center(), Face()) & (H > 0)
    return ifelse(active, fU, zero(grid))
end

#####
##### Evolution kernels: U first with the old V, then V with the new U
#####

@kernel function _split_explicit_barotropic_x_velocity!(transport_weight, grid, filled_halos, Δτ, η, U, V, Gᵁ, g, Ũ, Vʳ, coriolis, timestepper)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    Hᶠᶜ = x_column_depth(i, j, k_top, grid, filled_halos, η)
    ∂xᵣ = x_derivative_operator(filled_halos)
    fV  = barotropic_x_coriolis(i, j, k_top, grid, coriolis, V, Vʳ, Hᶠᶜ)

    @inbounds begin
        U[i, j, 1] += Δτ * (- g * Hᶠᶜ * ∂xᵣ(i, j, k_top, grid, η★, timestepper, η) + Gᵁ[i, j, 1] + fV)
        Ũ[i, j, 1] += transport_weight * U[i, j, 1]
    end
end

@kernel function _split_explicit_barotropic_y_velocity!(transport_weight, grid, filled_halos, Δτ, η, U, V, Gⱽ, g, Ṽ, Uʳ, coriolis, timestepper)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    Hᶜᶠ = y_column_depth(i, j, k_top, grid, filled_halos, η)
    ∂yᵣ = y_derivative_operator(filled_halos)
    fU  = barotropic_y_coriolis(i, j, k_top, grid, coriolis, U, Uʳ, Hᶜᶠ)

    @inbounds begin
        V[i, j, 1] += Δτ * (- g * Hᶜᶠ * ∂yᵣ(i, j, k_top, grid, η★, timestepper, η) + Gⱽ[i, j, 1] - fU)
        Ṽ[i, j, 1] += transport_weight * V[i, j, 1]
    end
end

#####
##### Substepping drivers
#####

iterate_split_explicit!(free_surface::CoriolisSplitExplicit, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, ::Val{Nsubsteps}, coriolis) where Nsubsteps =
    @apply_regionally iterate_split_explicit_in_halo!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, Val(Nsubsteps), coriolis)

function iterate_split_explicit_in_halo!(free_surface::CoriolisSplitExplicit, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, ::Val{Nsubsteps}, coriolis) where Nsubsteps
    arch = architecture(grid)

    η           = free_surface.displacement
    grid        = free_surface.displacement.grid
    state       = free_surface.filtered_state
    timestepper = free_surface.timestepper
    g           = free_surface.gravitational_acceleration
    parameters  = free_surface.kernel_parameters

    U, V    = free_surface.barotropic_velocities
    Uʳ, Vʳ  = timestepper.reference_transports
    η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅
    Ũ, Ṽ    = state.Ũ, state.Ṽ

    x_velocity_kernel!, _   = configure_kernel(arch, grid, parameters, _split_explicit_barotropic_x_velocity!)
    y_velocity_kernel!, _   = configure_kernel(arch, grid, parameters, _split_explicit_barotropic_y_velocity!)
    free_surface_kernel!, _ = configure_kernel(arch, grid, parameters, _split_explicit_free_surface!)

    U_args = (grid, Val(false), Δτᴮ, η, U, V, GUⁿ, g, Ũ, Vʳ, coriolis, timestepper)
    V_args = (grid, Val(false), Δτᴮ, η, U, V, GVⁿ, g, Ṽ, Uʳ, coriolis, timestepper)
    η_args = (grid, Val(false), Δτᴮ, η, U, V, F, clock, η̅, U̅, V̅, timestepper)

    GC.@preserve U_args V_args η_args begin
        converted_U_args = convert_to_device(arch, U_args)
        converted_V_args = convert_to_device(arch, V_args)
        converted_η_args = convert_to_device(arch, η_args)

        for substep in 1:Nsubsteps # unrolling stalls LLVM's SLP vectorizer on immersed grids
            @inbounds averaging_weight = weights[substep]
            @inbounds transport_weight = transport_weights[substep]

            x_velocity_kernel!(transport_weight, converted_U_args...)
            y_velocity_kernel!(transport_weight, converted_V_args...)
            free_surface_kernel!(averaging_weight, converted_η_args...)
        end
    end

    return nothing
end

function iterate_split_explicit!(free_surface::FillHaloCoriolisSplitExplicit, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, ::Val{Nsubsteps}, coriolis) where Nsubsteps
    η           = free_surface.displacement
    grid        = free_surface.displacement.grid
    arch        = architecture(grid)
    state       = free_surface.filtered_state
    timestepper = free_surface.timestepper
    g           = free_surface.gravitational_acceleration
    parameters  = free_surface.kernel_parameters

    U, V    = free_surface.barotropic_velocities
    Uʳ, Vʳ  = timestepper.reference_transports
    η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅
    Ũ, Ṽ    = state.Ũ, state.Ṽ

    @apply_regionally x_velocity_kernel!, _   = configure_kernel(arch, grid, parameters, _split_explicit_barotropic_x_velocity!)
    @apply_regionally y_velocity_kernel!, _   = configure_kernel(arch, grid, parameters, _split_explicit_barotropic_y_velocity!)
    @apply_regionally free_surface_kernel!, _ = configure_kernel(arch, grid, parameters, _split_explicit_free_surface!)

    U_args = (grid, Val(true), Δτᴮ, η, U, V, GUⁿ, g, Ũ, Vʳ, coriolis, timestepper)
    V_args = (grid, Val(true), Δτᴮ, η, U, V, GVⁿ, g, Ṽ, Uʳ, coriolis, timestepper)
    η_args = (grid, Val(true), Δτᴮ, η, U, V, F, clock, η̅, U̅, V̅, timestepper)

    barotropic_model_fields = (; U, V, η)

    substep_clock = (; time = clock.time, iteration = clock.iteration, stage = 0, last_stage_Δt = Δτᴮ)
    @apply_regionally U_halo_args = build_halo_fill_args(U, grid, substep_clock, barotropic_model_fields)
    @apply_regionally V_halo_args = build_halo_fill_args(V, grid, substep_clock, barotropic_model_fields)
    @apply_regionally η_halo_args = build_halo_fill_args(η, grid, substep_clock, barotropic_model_fields)

    only_local_halos = fill_only_local_halos(free_surface)

    GC.@preserve U_args V_args η_args U_halo_args V_halo_args η_halo_args begin
        @apply_regionally converted_U_args = convert_to_device(arch, U_args)
        @apply_regionally converted_V_args = convert_to_device(arch, V_args)
        @apply_regionally converted_η_args = convert_to_device(arch, η_args)
        @apply_regionally converted_U_halo_args = prepare_halo_fill_args(arch, U_halo_args, grid, free_surface)
        @apply_regionally converted_V_halo_args = prepare_halo_fill_args(arch, V_halo_args, grid, free_surface)
        @apply_regionally converted_η_halo_args = prepare_halo_fill_args(arch, η_halo_args, grid, free_surface)

        for substep in 1:Nsubsteps # unrolling stalls LLVM's SLP vectorizer on immersed grids
            @inbounds averaging_weight = weights[substep]
            @inbounds transport_weight = transport_weights[substep]

            maybe_distributed_fill_halo_regions!(arch, converted_η_halo_args...; only_local_halos)
            @apply_regionally apply_barotropic_kernel!(x_velocity_kernel!, transport_weight, converted_U_args)

            maybe_distributed_fill_halo_regions!(arch, converted_U_halo_args...; only_local_halos)
            @apply_regionally apply_barotropic_kernel!(y_velocity_kernel!, transport_weight, converted_V_args)

            maybe_distributed_fill_halo_regions!(arch, converted_V_halo_args...; only_local_halos)
            @apply_regionally apply_barotropic_kernel!(free_surface_kernel!, averaging_weight, converted_η_args)
        end
    end

    return nothing
end
