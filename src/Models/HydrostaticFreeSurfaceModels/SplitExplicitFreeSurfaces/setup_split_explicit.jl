using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, RungeKutta3TimeStepper

# `initialize_free_surface!` is called at the beginning of the simulation to initialize the free surface state
# from the initial velocity conditions.
function initialize_free_surface!(sefs::SplitExplicitFreeSurface, grid, velocities)
    barotropic_velocities = sefs.barotropic_velocities
    @apply_regionally compute_barotropic_mode!(barotropic_velocities.U, barotropic_velocities.V, grid, velocities.u, velocities.v)
    fill_halo_regions!((barotropic_velocities.U, barotropic_velocities.V))
    return nothing
end

# `initialize_free_surface_state!` is called at the beginning of the substepping to 
# reset the filtered state to zero and reinitialize the state from the filtered state.
function initialize_free_surface_state!(filtered_state, η, velocities, timestepper)

    initialize_free_surface_timestepper!(timestepper, η, velocities)

    fill!(filtered_state.η, 0)
    fill!(filtered_state.U, 0)
    fill!(filtered_state.V, 0)

    return nothing
end

# Calculate RHS for the barotropic time step.
@kernel function _compute_integrated_ab2_tendencies!(Gᵁ, Gⱽ, grid, ::Nothing, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)
    i, j  = @index(Global, NTuple)

    @inbounds Gᵁ[i, j, 1] = Δzᶠᶜᶜ(i, j, 1, grid) * ab2_step_Gu(i, j, 1, grid, Gu⁻, Guⁿ, χ)
    @inbounds Gⱽ[i, j, 1] = Δzᶜᶠᶜ(i, j, 1, grid) * ab2_step_Gv(i, j, 1, grid, Gv⁻, Gvⁿ, χ)

    for k in 2:grid.Nz
        @inbounds Gᵁ[i, j, 1] += Δzᶠᶜᶜ(i, j, k, grid) * ab2_step_Gu(i, j, k, grid, Gu⁻, Guⁿ, χ)
        @inbounds Gⱽ[i, j, 1] += Δzᶜᶠᶜ(i, j, k, grid) * ab2_step_Gv(i, j, k, grid, Gv⁻, Gvⁿ, χ)
    end
end

# Calculate RHS for the barotropic time step.q
@kernel function _compute_integrated_ab2_tendencies!(Gᵁ, Gⱽ, grid, active_cells_map, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)
    idx = @index(Global, Linear)
    i, j = active_linear_index_to_tuple(idx, active_cells_map)

    @inbounds Gᵁ[i, j, 1] = Δzᶠᶜᶜ(i, j, 1, grid) * ab2_step_Gu(i, j, 1, grid, Gu⁻, Guⁿ, χ)
    @inbounds Gⱽ[i, j, 1] = Δzᶜᶠᶜ(i, j, 1, grid) * ab2_step_Gv(i, j, 1, grid, Gv⁻, Gvⁿ, χ)

    for k in 2:grid.Nz
        @inbounds Gᵁ[i, j, 1] += Δzᶠᶜᶜ(i, j, k, grid) * ab2_step_Gu(i, j, k, grid, Gu⁻, Guⁿ, χ)
        @inbounds Gⱽ[i, j, 1] += Δzᶜᶠᶜ(i, j, k, grid) * ab2_step_Gv(i, j, k, grid, Gv⁻, Gvⁿ, χ)
    end
end

@inline ab2_step_Gu(i, j, k, grid, G⁻, Gⁿ, χ::FT) where FT =
    @inbounds ifelse(peripheral_node(i, j, k, grid, f, c, c), zero(grid), (convert(FT, 1.5) + χ) *  Gⁿ[i, j, k] - G⁻[i, j, k] * (convert(FT, 0.5) + χ))

@inline ab2_step_Gv(i, j, k, grid, G⁻, Gⁿ, χ::FT) where FT =
    @inbounds ifelse(peripheral_node(i, j, k, grid, c, f, c), zero(grid), (convert(FT, 1.5) + χ) *  Gⁿ[i, j, k] - G⁻[i, j, k] * (convert(FT, 0.5) + χ))

# Calculate RHS for the barotropic time step.
@kernel function _compute_integrated_tendencies!(Gᵁ, Gⱽ, grid, Guⁿ, Gvⁿ)
    i, j  = @index(Global, NTuple)
    k_top = grid.Nz + 1

    @inbounds Gᵁ[i, j, k_top-1] = Δzᶠᶜᶜ(i, j, 1, grid) * ifelse(peripheral_node(i, j, 1, grid, f, c, c), zero(grid), Guⁿ[i, j, 1])
    @inbounds Gⱽ[i, j, k_top-1] = Δzᶜᶠᶜ(i, j, 1, grid) * ifelse(peripheral_node(i, j, 1, grid, c, f, c), zero(grid), Gvⁿ[i, j, 1])

    for k in 2:grid.Nz	
        @inbounds Gᵁ[i, j, k_top-1] += Δzᶠᶜᶜ(i, j, k, grid) * ifelse(peripheral_node(i, j, k, grid, f, c, c), zero(grid), Guⁿ[i, j, k])
        @inbounds Gⱽ[i, j, k_top-1] += Δzᶜᶠᶜ(i, j, k, grid) * ifelse(peripheral_node(i, j, k, grid, c, f, c), zero(grid), Gvⁿ[i, j, k])
    end	
end

@inline function ab2_split_explicit_forcing!(GUⁿ, GVⁿ, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)
    active_cells_map = retrieve_surface_active_cells_map(grid)

    launch!(architecture(grid), grid, :xy, _compute_integrated_ab2_tendencies!, GUⁿ, GVⁿ, grid,
            active_cells_map, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ; active_cells_map)

    return nothing
end

@inline function vertical_integral(i, j, grid, Gⁿ, ℓx, ℓy, ℓz)
    G = Δz(i, j, 1, grid, ℓx, ℓy, ℓz) * ifelse(peripheral_node(i, j, 1, grid, ℓx, ℓy, ℓz), zero(grid), Gⁿ[i, j, 1])
    
    for k in 2:grid.Nz	
        @inbounds G += Δz(i, j, k, grid, ℓx, ℓy, ℓz) * ifelse(peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz), zero(grid), Gⁿ[i, j, k])
    end
    
    return G
end

@kernel function _compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, grid, Guⁿ, Gvⁿ, ::Val{1})
    i, j = @index(Global, NTuple)

    @inbounds GUⁿ[i, j, 1] = vertical_integral(i, j, grid, Guⁿ, f, c, c)
    @inbounds GVⁿ[i, j, 1] = vertical_integral(i, j, grid, Gvⁿ, c, f, c)

    @inbounds GU⁻[i, j, 1] = GUⁿ[i, j, 1]
    @inbounds GV⁻[i, j, 1] = GVⁿ[i, j, 1]
end

@kernel function _compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, grid, Guⁿ, Gvⁿ, ::Val{2})
    i, j = @index(Global, NTuple)

    @inbounds GUⁿ[i, j, 1] = vertical_integral(i, j, grid, Guⁿ, f, c, c)
    @inbounds GVⁿ[i, j, 1] = vertical_integral(i, j, grid, Gvⁿ, c, f, c)

    @inbounds GU⁻[i, j, 1] = 1 // 6 * GUⁿ[i, j, 1] + 1  // 6 * GU⁻[i, j, 1]
    @inbounds GV⁻[i, j, 1] = 1 // 6 * GVⁿ[i, j, 1] + 1  // 6 * GU⁻[i, j, 1]
end

@kernel function _compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, grid, Guⁿ, Gvⁿ, ::Val{3})
    i, j = @index(Global, NTuple)

    @inbounds GUⁿ[i, j, 1] = vertical_integral(i, j, grid, Guⁿ, f, c, c)
    @inbounds GVⁿ[i, j, 1] = vertical_integral(i, j, grid, Gvⁿ, c, f, c)

    @inbounds GUⁿ[i, j, 1] = 2 // 3 * GUⁿ[i, j, 1] + GU⁻[i, j, 1]
    @inbounds GVⁿ[i, j, 1] = 2 // 3 * GVⁿ[i, j, 1] + GV⁻[i, j, 1]
end

rk3_split_explicit_forcing!(GUⁿ, GVⁿ, GU⁻, GV⁻, grid, Guⁿ, Gvⁿ, stage) =
    launch!(architecture(grid), grid, :xy, _compute_integrated_rk3_tendencies!, GUⁿ, GVⁿ, GU⁻, GV⁻, grid, Guⁿ, Gvⁿ, stage)

split_explicit_forcing!(GUⁿ, GVⁿ, GU⁻, GV⁻, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, timestepper::QuasiAdamsBashforth2TimeStepper, stage) = 
    ab2_split_explicit_forcing!(GUⁿ, GVⁿ, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, timestepper.χ)

split_explicit_forcing!(GUⁿ, GVⁿ, GU⁻, GV⁻, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, ::RungeKutta3TimeStepper, stage)  = 
    rk3_split_explicit_forcing!(GUⁿ, GVⁿ, GU⁻, GV⁻, grid, Guⁿ, Gvⁿ, Val(stage))

# Setting up the RHS for the barotropic step (tendencies of the barotropic velocity components)
# This function is called after `calculate_tendency` and before `ab2_step_velocities!`
function setup_free_surface!(model, free_surface::SplitExplicitFreeSurface, timestepper, stage)
    
    # we start the time integration of η from the average ηⁿ     
    Gu⁻ = model.timestepper.G⁻.u
    Gv⁻ = model.timestepper.G⁻.v
    Guⁿ = model.timestepper.Gⁿ.u
    Gvⁿ = model.timestepper.Gⁿ.v

    GU⁻ = model.timestepper.Gⁿ.U
    GV⁻ = model.timestepper.Gⁿ.V
    GUⁿ = model.timestepper.Gⁿ.U
    GVⁿ = model.timestepper.Gⁿ.V
    
    @apply_regionally split_explicit_forcing!(GUⁿ, GVⁿ, GU⁻, GV⁻, model.grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, timestepper, stage)

    initialize_free_surface_state!(free_surface.state, free_surface.η, settings.timestepper, stage)

    fields_to_fill = (GUⁿ, GVⁿ)
    fill_halo_regions!(fields_to_fill; async = true)

    return nothing
end