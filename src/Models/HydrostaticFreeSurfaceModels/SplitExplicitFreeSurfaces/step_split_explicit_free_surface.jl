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

@inline split_explicit_covariant_xface_source_value(i, j, k, grid, filled_halos, timestepper, U, V) =
    U★(i, j, k, grid, timestepper, U)

@inline split_explicit_covariant_yface_source_value(i, j, k, grid, filled_halos, timestepper, U, V) =
    U★(i, j, k, grid, timestepper, V)

@inline inside_octahealpix_horizontal_domain(i, j, grid) =
    (i >= 1) & (i <= grid.Nx) & (j >= 1) & (j <= grid.Ny)

@inline split_explicit_covariant_xface_source_value(i, j, k,
                                                    grid::SphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:OctaHEALPixConnectivity},
                                                    ::Val{true}, timestepper, U, V) =
    U★(i, j, k, grid, timestepper, U)

@inline split_explicit_covariant_yface_source_value(i, j, k,
                                                    grid::SphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:OctaHEALPixConnectivity},
                                                    ::Val{true}, timestepper, U, V) =
    U★(i, j, k, grid, timestepper, V)

@inline function split_explicit_covariant_xface_source_value(i, j, k,
                                                             grid::SphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:OctaHEALPixConnectivity},
                                                             filled_halos, timestepper, U, V)
    inside = inside_octahealpix_horizontal_domain(i, j, grid)
    source_kind, source_i, source_j, sign,
    _, _, _, _ =
        octahealpix_vector_halo_source_pair(i, j, grid.Nx, grid.Ny, grid.connectivity, Val(:covariant))

    safe_i = ifelse(inside, i, source_i)
    safe_j = ifelse(inside, j, source_j)
    source_u = U★(safe_i, safe_j, k, grid, timestepper, U)
    source_v = U★(safe_i, safe_j, k, grid, timestepper, V)
    halo_value = ifelse(source_kind == 1, sign * source_u, sign * source_v)

    return ifelse(inside, source_u, halo_value)
end

@inline function split_explicit_covariant_yface_source_value(i, j, k,
                                                             grid::SphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:OctaHEALPixConnectivity},
                                                             filled_halos, timestepper, U, V)
    inside = inside_octahealpix_horizontal_domain(i, j, grid)
    _, _, _, _,
    source_kind, source_i, source_j, sign =
        octahealpix_vector_halo_source_pair(i, j, grid.Nx, grid.Ny, grid.connectivity, Val(:covariant))

    safe_i = ifelse(inside, i, source_i)
    safe_j = ifelse(inside, j, source_j)
    source_u = U★(safe_i, safe_j, k, grid, timestepper, U)
    source_v = U★(safe_i, safe_j, k, grid, timestepper, V)
    halo_value = ifelse(source_kind == 1, sign * source_u, sign * source_v)

    return ifelse(inside, source_v, halo_value)
end

@inline split_explicit_surface_source_value(i, j, k, grid, filled_halos, timestepper, η) =
    η★(i, j, k, grid, timestepper, η)

@inline function split_explicit_surface_source_value(i, j, k,
                                                     grid::SphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:OctaHEALPixConnectivity},
                                                     ::Val{false}, timestepper, η)
    inside = inside_octahealpix_horizontal_domain(i, j, grid)
    source_ring = Oceananigans.Grids.octahealpix_halo_source_ring_index(i, j, grid.Nx, grid.Ny, grid.connectivity)
    source_i = grid.connectivity.ring_to_i[source_ring]
    source_j = grid.connectivity.ring_to_j[source_ring]

    safe_i = ifelse(inside, i, source_i)
    safe_j = ifelse(inside, j, source_j)
    return η★(safe_i, safe_j, k, grid, timestepper, η)
end

@inline function split_explicit_barotropic_pressure_gradient_u(i, j, k_top, grid, filled_halos, timestepper, η)
    ∂xᵣ = x_derivative_operator(filled_halos)
    return ∂xᵣ(i, j, k_top, grid, η★, timestepper, η)
end

@inline function split_explicit_barotropic_pressure_gradient_v(i, j, k_top, grid, filled_halos, timestepper, η)
    ∂yᵣ = y_derivative_operator(filled_halos)
    return ∂yᵣ(i, j, k_top, grid, η★, timestepper, η)
end

@inline function split_explicit_barotropic_pressure_gradient_u(i, j, k_top,
                                                               grid::SphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:OctaHEALPixConnectivity},
                                                               ::Val{false}, timestepper, η)
    return ∂xᵣᶠᶜᶠ(i, j, k_top, grid,
                  split_explicit_surface_source_value,
                  Val(false), timestepper, η)
end

@inline function split_explicit_barotropic_pressure_gradient_v(i, j, k_top,
                                                               grid::SphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:OctaHEALPixConnectivity},
                                                               ::Val{false}, timestepper, η)
    return ∂yᵣᶜᶠᶠ(i, j, k_top, grid,
                  split_explicit_surface_source_value,
                  Val(false), timestepper, η)
end

@inline function split_explicit_barotropic_contravariant_flux_u(i, j, k,
                                                                grid::SphericalShellGrid,
                                                                filled_halos,
                                                                timestepper,
                                                                U, V)
    Ū = split_explicit_covariant_xface_source_value(i, j, k, grid, filled_halos, timestepper, U, V)
    V₁ = split_explicit_covariant_yface_source_value(i - 1, j,     k, grid, filled_halos, timestepper, U, V)
    V₂ = split_explicit_covariant_yface_source_value(i,     j,     k, grid, filled_halos, timestepper, U, V)
    V₃ = split_explicit_covariant_yface_source_value(i - 1, j + 1, k, grid, filled_halos, timestepper, U, V)
    V₄ = split_explicit_covariant_yface_source_value(i,     j + 1, k, grid, filled_halos, timestepper, U, V)

    quarter = convert(typeof(Ū), 1//4)
    V̄ = quarter * (V₁ + V₂ + V₃ + V₄)

    return G¹¹ᶠᶜᶜ(i, j, k, grid) * Ū +
           G¹²ᶠᶜᶜ(i, j, k, grid) * V̄
end

@inline function split_explicit_barotropic_contravariant_flux_v(i, j, k,
                                                                grid::SphericalShellGrid,
                                                                filled_halos,
                                                                timestepper,
                                                                U, V)
    U₁ = split_explicit_covariant_xface_source_value(i,     j - 1, k, grid, filled_halos, timestepper, U, V)
    U₂ = split_explicit_covariant_xface_source_value(i + 1, j - 1, k, grid, filled_halos, timestepper, U, V)
    U₃ = split_explicit_covariant_xface_source_value(i,     j,     k, grid, filled_halos, timestepper, U, V)
    U₄ = split_explicit_covariant_xface_source_value(i + 1, j,     k, grid, filled_halos, timestepper, U, V)
    V̄ = split_explicit_covariant_yface_source_value(i, j, k, grid, filled_halos, timestepper, U, V)

    quarter = convert(typeof(V̄), 1//4)
    Ū = quarter * (U₁ + U₂ + U₃ + U₄)

    return G²¹ᶜᶠᶜ(i, j, k, grid) * Ū +
           G²²ᶜᶠᶜ(i, j, k, grid) * V̄
end

@inline split_explicit_barotropic_transport_flux_u(i, j, k,
                                                   grid::SphericalShellGrid,
                                                   filled_halos,
                                                   timestepper,
                                                   U, V) =
    Oceananigans.Operators.transverse_computational_width_uᶠᶜᶜ(i, j, k, grid) *
    split_explicit_barotropic_contravariant_flux_u(i, j, k, grid, filled_halos, timestepper, U, V)

@inline split_explicit_barotropic_transport_flux_v(i, j, k,
                                                   grid::SphericalShellGrid,
                                                   filled_halos,
                                                   timestepper,
                                                   U, V) =
    Oceananigans.Operators.transverse_computational_width_vᶜᶠᶜ(i, j, k, grid) *
    split_explicit_barotropic_contravariant_flux_v(i, j, k, grid, filled_halos, timestepper, U, V)

@inline function split_explicit_free_surface_barotropic_divergence(i, j, k_top, grid, filled_halos, timestepper, U, V)
    δx = x_difference_operator(filled_halos)
    δy = y_difference_operator(filled_halos)

    return (δx(i, j, grid.Nz, grid, Δy_qᶠᶜᶠ, U★, timestepper, U) +
            δy(i, j, grid.Nz, grid, Δx_qᶜᶠᶠ, U★, timestepper, V)) *
           Az⁻¹ᶜᶜᶠ(i, j, k_top, grid)
end

@inline function split_explicit_free_surface_barotropic_divergence(i, j, k_top,
                                                                   grid::SphericalShellGrid,
                                                                   filled_halos,
                                                                   timestepper,
                                                                   U, V)
    δx = x_difference_operator(filled_halos)
    δy = y_difference_operator(filled_halos)

    return (δx(i, j, grid.Nz, grid,
               split_explicit_barotropic_transport_flux_u,
               filled_halos, timestepper, U, V) +
            δy(i, j, grid.Nz, grid,
               split_explicit_barotropic_transport_flux_v,
               filled_halos, timestepper, U, V)) *
           Az⁻¹ᶜᶜᶠ(i, j, k_top, grid)
end

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
    ∂x_η = split_explicit_barotropic_pressure_gradient_u(i, j, k_top, grid, filled_halos, timestepper, η)
    ∂y_η = split_explicit_barotropic_pressure_gradient_v(i, j, k_top, grid, filled_halos, timestepper, η)

    # ∂τ(U) = - ∇η + G
    # On OctaHEALPix, η-gradients must use the non-orthogonal covariant surface-gradient path.
    @inbounds begin
        U[i, j, 1] += Δτ * (- g * Hᶠᶜ * ∂x_η + Gᵁ[i, j, 1])
        V[i, j, 1] += Δτ * (- g * Hᶜᶠ * ∂y_η + Gⱽ[i, j, 1])

        # Averaging the transport
        Ũ[i, j, 1] += transport_weight * U[i, j, 1]
        Ṽ[i, j, 1] += transport_weight * V[i, j, 1]
    end
end

@kernel function _split_explicit_free_surface!(averaging_weight, grid, filled_halos, Δτ, η, U, V, F, clock, η̅, U̅, V̅, timestepper)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    cache_previous_free_surface!(timestepper, i, j, k_top, η)

    δh_U = split_explicit_free_surface_barotropic_divergence(i, j, k_top, grid, filled_halos, timestepper, U, V)

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

    GC.@preserve U_args η_args begin
        # We need to perform ~50 time-steps which means launching ~100 very small kernels: we are limited by latency of
        # argument conversion to GPU-compatible values. To alleviate this penalty we convert first and then we substep!
        @apply_regionally converted_U_args = convert_to_device(arch, U_args)
        @apply_regionally converted_η_args = convert_to_device(arch, η_args)

        @unroll for substep in 1:Nsubsteps
            @inbounds averaging_weight = weights[substep]
            @inbounds transport_weight = transport_weights[substep]

            fill_halo_regions!(η)
            @apply_regionally apply_barotropic_kernel!(velocity_kernel!, transport_weight, converted_U_args)

            fill_halo_regions!((U, V))
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

function step_free_surface!(free_surface::SplitExplicitFreeSurface, model, baroclinic_timestepper, Δt)
    Oceananigans.Models.update_model_field_time_series!(model, model.clock)

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
    fill_halo_regions!((filtered_state.Ũ, filtered_state.Ṽ); async=true)
    fill_halo_regions!((U, V); async=true)
    fill_halo_regions!(η; async=true)

    return nothing
end
