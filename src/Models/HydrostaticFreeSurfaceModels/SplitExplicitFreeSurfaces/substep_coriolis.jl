using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Coriolis: ESC, ENC, AESC, AENC, TS, 𝒯⁺⁺, 𝒯⁻⁺, 𝒯⁺⁻, 𝒯⁻⁻
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ

#####
##### Barotropic Coriolis inside the substeps (`CoriolisForwardBackwardScheme`)
#####

const CoriolisSplitExplicit = SplitExplicitFreeSurface{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:CoriolisForwardBackwardScheme}

substep_coriolis_args(timestepper, model) = ()
substep_coriolis_args(::CoriolisForwardBackwardScheme, model) = (model.coriolis,)

#####
##### Reference transports: the barotropic transports whose Coriolis force is already in `Gᵁ` and `Gⱽ`
#####

function initialize_free_surface_state!(free_surface, baroclinic_timestepper::QuasiAdamsBashforth2TimeStepper, timestepper::CoriolisForwardBackwardScheme)
    U, V = free_surface.barotropic_velocities
    Uʳ, Vʳ = timestepper.reference_transports
    U⁻, V⁻ = timestepper.previous_transports

    FT = eltype(U)
    α = convert(FT, 3/2) + baroclinic_timestepper.χ
    β = convert(FT, 1/2) + baroclinic_timestepper.χ

    parent(Uʳ) .= α .* parent(U) .- β .* parent(U⁻)
    parent(Vʳ) .= α .* parent(V) .- β .* parent(V⁻)
    parent(U⁻) .= parent(U)
    parent(V⁻) .= parent(V)

    return invoke(initialize_free_surface_state!, Tuple{Any, Any, Any}, free_surface, baroclinic_timestepper, timestepper)
end

# Before restarting from the state at tⁿ, `U` and `V` hold the transports of the previous stage
function initialize_free_surface_state!(free_surface, baroclinic_timestepper::SplitRungeKuttaTimeStepper, timestepper::CoriolisForwardBackwardScheme)
    parent(timestepper.reference_transports.U) .= parent(free_surface.barotropic_velocities.U)
    parent(timestepper.reference_transports.V) .= parent(free_surface.barotropic_velocities.V)
    return invoke(initialize_free_surface_state!, Tuple{Any, SplitRungeKuttaTimeStepper, Any}, free_surface, baroclinic_timestepper, timestepper)
end

#####
##### Barotropic Coriolis operators: the stencils of `x_f_cross_U` and `y_f_cross_U` applied to the transport deviation
#####

@inline horizontal_peripheral_node(i, j, grid, ℓx, ℓy) = peripheral_node(i, j, 1, grid, ℓx, ℓy, Center())
@inline horizontal_peripheral_node(i, j, grid::ImmersedBoundaryGrid, ℓx, ℓy) = horizontal_peripheral_node(i, j, grid.underlying_grid, ℓx, ℓy)

# Periodic halos are not refilled between substeps, so wrap the stencil indices like the topology-aware differences
@inline wrap_index(i, N, topo) = i
@inline wrap_index(i, N, ::Type{Periodic}) = ifelse(i < 1, i + N, ifelse(i > N, i - N, i))

@inline function Δx_δVᶜᶠᶠ(i, j, k, grid, V, Vʳ)
    i′ = wrap_index(i, grid.Nx, topology(grid, 1))
    j′ = wrap_index(j, grid.Ny, topology(grid, 2))
    return @inbounds Δxᶜᶠᶠ(i, j, k, grid) * (V[i′, j′, 1] - Vʳ[i′, j′, 1])
end

@inline function Δy_δUᶠᶜᶠ(i, j, k, grid, U, Uʳ)
    i′ = wrap_index(i, grid.Nx, topology(grid, 1))
    j′ = wrap_index(j, grid.Ny, topology(grid, 2))
    return @inbounds Δyᶠᶜᶠ(i, j, k, grid) * (U[i′, j′, 1] - Uʳ[i′, j′, 1])
end

# Wet columns, counted by the active-weighted schemes in place of active cells
@inline wet_columnᶜᶠ(i, j, k, grid) = !peripheral_node(i, j, grid.Nz, grid, Center(), Face(), Center())
@inline wet_columnᶠᶜ(i, j, k, grid) = !peripheral_node(i, j, grid.Nz, grid, Face(), Center(), Center())

@inline x_f_cross_δU(i, j, k, grid, ::Nothing, V, Vʳ) = zero(grid)
@inline y_f_cross_δU(i, j, k, grid, ::Nothing, U, Uʳ) = zero(grid)

@inline x_f_cross_δU(i, j, k, grid, coriolis::ESC, V, Vʳ) = - ℑyᵃᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * ℑxyᶠᶜᵃ(i, j, k, grid, Δx_δVᶜᶠᶠ, V, Vʳ) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
@inline y_f_cross_δU(i, j, k, grid, coriolis::ESC, U, Uʳ) = + ℑxᶜᵃᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * ℑxyᶜᶠᵃ(i, j, k, grid, Δy_δUᶠᶜᶠ, U, Uʳ) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)

@inline f_ℑx_Δx_δVᶠᶠᶠ(i, j, k, grid, coriolis, V, Vʳ) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_δVᶜᶠᶠ, V, Vʳ)
@inline f_ℑy_Δy_δUᶠᶠᶠ(i, j, k, grid, coriolis, U, Uʳ) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_δUᶠᶜᶠ, U, Uʳ)

@inline x_f_cross_δU(i, j, k, grid, coriolis::ENC, V, Vʳ) = - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_Δx_δVᶠᶠᶠ, coriolis, V, Vʳ) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
@inline y_f_cross_δU(i, j, k, grid, coriolis::ENC, U, Uʳ) = + ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_Δy_δUᶠᶠᶠ, coriolis, U, Uʳ) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)

@inline function x_f_cross_δU(i, j, k, grid, coriolis::AESC, V, Vʳ)
    active_columns = ℑxyᶠᶜᵃ(i, j, k, grid, wet_columnᶜᶠ)
    result = - ℑyᵃᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * ℑxyᶠᶜᵃ(i, j, k, grid, Δx_δVᶜᶠᶠ, V, Vʳ)
    return ifelse(active_columns == 0, zero(grid), result / active_columns) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
end

@inline function y_f_cross_δU(i, j, k, grid, coriolis::AESC, U, Uʳ)
    active_columns = ℑxyᶜᶠᵃ(i, j, k, grid, wet_columnᶠᶜ)
    result = ℑxᶜᵃᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * ℑxyᶜᶠᵃ(i, j, k, grid, Δy_δUᶠᶜᶠ, U, Uʳ)
    return ifelse(active_columns == 0, zero(grid), result / active_columns) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)
end

@inline function x_f_cross_δU(i, j, k, grid, coriolis::AENC, V, Vʳ)
    active_columns = ℑxyᶠᶜᵃ(i, j, k, grid, wet_columnᶜᶠ)
    result = - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_Δx_δVᶠᶠᶠ, coriolis, V, Vʳ) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
    return ifelse(active_columns == 0, zero(grid), result / active_columns)
end

@inline function y_f_cross_δU(i, j, k, grid, coriolis::AENC, U, Uʳ)
    active_columns = ℑxyᶜᶠᵃ(i, j, k, grid, wet_columnᶠᶜ)
    result = ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_Δy_δUᶠᶠᶠ, coriolis, U, Uʳ) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)
    return ifelse(active_columns == 0, zero(grid), result / active_columns)
end

@inline x_f_cross_δU(i, j, k, grid, coriolis::TS, V, Vʳ) =
    - Δx⁻¹ᶠᶜᶠ(i, j, k, grid) / 12 * (𝒯⁺⁺(i-1, j, k, grid, coriolis) * Δx_δVᶜᶠᶠ(i-1, j+1, k, grid, V, Vʳ) +
                                      𝒯⁻⁻(i,   j, k, grid, coriolis) * Δx_δVᶜᶠᶠ(i,   j,   k, grid, V, Vʳ) +
                                      𝒯⁺⁻(i-1, j, k, grid, coriolis) * Δx_δVᶜᶠᶠ(i-1, j,   k, grid, V, Vʳ) +
                                      𝒯⁻⁺(i,   j, k, grid, coriolis) * Δx_δVᶜᶠᶠ(i,   j+1, k, grid, V, Vʳ))

@inline y_f_cross_δU(i, j, k, grid, coriolis::TS, U, Uʳ) =
    + Δy⁻¹ᶜᶠᶠ(i, j, k, grid) / 12 * (𝒯⁻⁻(i, j,   k, grid, coriolis) * Δy_δUᶠᶜᶠ(i,   j,   k, grid, U, Uʳ) +
                                      𝒯⁺⁺(i, j-1, k, grid, coriolis) * Δy_δUᶠᶜᶠ(i+1, j-1, k, grid, U, Uʳ) +
                                      𝒯⁻⁺(i, j-1, k, grid, coriolis) * Δy_δUᶠᶜᶠ(i,   j-1, k, grid, U, Uʳ) +
                                      𝒯⁺⁻(i, j,   k, grid, coriolis) * Δy_δUᶠᶜᶠ(i+1, j,   k, grid, U, Uʳ))

@inline function barotropic_x_coriolis(i, j, k, grid, coriolis, V, Vʳ, H)
    active = !horizontal_peripheral_node(i, j, grid, Face(), Center()) & (H > 0)
    return ifelse(active, x_f_cross_δU(i, j, k, grid, coriolis, V, Vʳ), zero(grid))
end

@inline function barotropic_y_coriolis(i, j, k, grid, coriolis, U, Uʳ, H)
    active = !horizontal_peripheral_node(i, j, grid, Center(), Face()) & (H > 0)
    return ifelse(active, y_f_cross_δU(i, j, k, grid, coriolis, U, Uʳ), zero(grid))
end

#####
##### Evolution kernels: U first with the old V, then V with the new U
#####

@kernel function _split_explicit_barotropic_x_velocity!(transport_weight, grid, filled_halos, Δτ, η, U, V, Gᵁ, g, Ũ, Vʳ, coriolis, timestepper)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    Hᶠᶜ = x_column_depth(i, j, k_top, grid, filled_halos, η)
    ∂xᵣ = x_derivative_operator(filled_halos)
    Cᵁ = barotropic_x_coriolis(i, j, k_top, grid, coriolis, V, Vʳ, Hᶠᶜ)

    @inbounds begin
        U[i, j, 1] += Δτ * (- g * Hᶠᶜ * ∂xᵣ(i, j, k_top, grid, η★, timestepper, η) + Gᵁ[i, j, 1] - Cᵁ)
        Ũ[i, j, 1] += transport_weight * U[i, j, 1]
    end
end

@kernel function _split_explicit_barotropic_y_velocity!(transport_weight, grid, filled_halos, Δτ, η, U, V, Gⱽ, g, Ṽ, Uʳ, coriolis, timestepper)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    Hᶜᶠ = y_column_depth(i, j, k_top, grid, filled_halos, η)
    ∂yᵣ = y_derivative_operator(filled_halos)
    Cⱽ = barotropic_y_coriolis(i, j, k_top, grid, coriolis, U, Uʳ, Hᶜᶠ)

    @inbounds begin
        V[i, j, 1] += Δτ * (- g * Hᶜᶠ * ∂yᵣ(i, j, k_top, grid, η★, timestepper, η) + Gⱽ[i, j, 1] - Cⱽ)
        Ṽ[i, j, 1] += transport_weight * V[i, j, 1]
    end
end

#####
##### Substepping driver, for both extended halos and halo filling
#####

@inline fills_halos(::SplitExplicitFreeSurface{ExtendedHalos}) = Val(false)
@inline fills_halos(::FillHaloSplitExplicit) = Val(true)

# Extended halos are refreshed all at once, only when the substeps outnumber them
@inline substep_exchange_halos!(::Val{true}, fields, exchange) = nothing

function substep_exchange_halos!(::Val{false}, fields, exchange)
    if exchange
        fill_halo_regions!(fields)
    end
    return nothing
end

@inline substep_fill_halos!(::Val{false}, arch, args, clock, only_local_halos) = nothing
@inline substep_fill_halos!(::Val{true},  arch, args, clock, only_local_halos) = maybe_distributed_fill_halo_regions!(arch, args[1:end-1]..., clock, args[end]; only_local_halos)

iterate_split_explicit!(free_surface::CoriolisSplitExplicit, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, ::Val{Nsubsteps}, coriolis) where Nsubsteps =
    @apply_regionally iterate_split_explicit_with_coriolis!(free_surface, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, Val(Nsubsteps), coriolis)

function iterate_split_explicit_with_coriolis!(free_surface, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, ::Val{Nsubsteps}, coriolis) where Nsubsteps
    η           = free_surface.displacement
    grid        = free_surface.displacement.grid
    arch        = architecture(grid)
    state       = free_surface.filtered_state
    timestepper = free_surface.timestepper
    g           = free_surface.gravitational_acceleration
    filled      = fills_halos(free_surface)

    U, V    = free_surface.barotropic_velocities
    Uʳ, Vʳ  = timestepper.reference_transports
    η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅
    Ũ, Ṽ    = state.Ũ, state.Ṽ

    workspec = Utils.possibly_load_active_cells_map(grid, free_surface.kernel_parameters, false)
    x_velocity_kernel!, _   = configure_kernel(arch, grid, workspec, _split_explicit_barotropic_x_velocity!)
    y_velocity_kernel!, _   = configure_kernel(arch, grid, workspec, _split_explicit_barotropic_y_velocity!)
    free_surface_kernel!, _ = configure_kernel(arch, grid, workspec, _split_explicit_free_surface!)

    U_args = (grid, filled, Δτᴮ, η, U, V, GUⁿ, g, Ũ, Vʳ, coriolis, timestepper)
    V_args = (grid, filled, Δτᴮ, η, U, V, GVⁿ, g, Ṽ, Uʳ, coriolis, timestepper)
    η_args = (grid, filled, Δτᴮ, η, U, V, F, clock, η̅, U̅, V̅, timestepper)

    barotropic_model_fields = (; U, V, η)
    Uh, Vh, ηh = map(field -> build_halo_fill_args(arch, field, grid, barotropic_model_fields), (U, V, η))
    exchange_interval = halo_exchange_interval(free_surface, grid, Nsubsteps)

    GC.@preserve U_args V_args η_args begin
        converted_U_args = convert_to_device(arch, U_args)
        converted_V_args = convert_to_device(arch, V_args)
        converted_η_args = convert_to_device(arch, η_args)
        face_pins = configure_face_pins(arch, grid, U, V, free_surface.boundary_transport)

        for substep in 1:Nsubsteps # unrolling stalls LLVM's SLP vectorizer on immersed grids
            @inbounds averaging_weight = weights[substep]
            @inbounds transport_weight = transport_weights[substep]

            substep_clock = (; time = clock.time + (substep - 1) * Δτᴮ, iteration = clock.iteration, stage = 0, last_stage_Δt = Δτᴮ)
            exchange = substep > 1 && (substep - 1) % exchange_interval == 0
            only_local_halos = !exchange

            substep_exchange_halos!(filled, (U, V, η), exchange)
            substep_fill_halos!(filled, arch, ηh, substep_clock, only_local_halos)
            x_velocity_kernel!(transport_weight, converted_U_args...)
            substep_fill_halos!(filled, arch, Uh, substep_clock, only_local_halos)
            y_velocity_kernel!(transport_weight, converted_V_args...)
            substep_fill_halos!(filled, arch, Vh, substep_clock, only_local_halos)
            pin_barotropic_faces!(face_pins)
            free_surface_kernel!(averaging_weight, converted_η_args...)
        end
    end

    return nothing
end
