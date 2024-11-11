using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, SplitRungeKutta3TimeStepper

#####
##### Initialize Free Surface state
#####

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
function initialize_free_surface_state!(free_surface, baroclinic_timestepper, timestepper, stage)

    η = free_surface.η
    U, V = free_surface.barotropic_velocities

    initialize_free_surface_timestepper!(timestepper, η, U, V)

    fill!(filtered_state.η, 0)
    fill!(filtered_state.U, 0)
    fill!(filtered_state.V, 0)

    return nothing
end

# At the first stage we reset the velocities and perform the complete substepping from n to n+1
function initialize_free_surface_state!(free_surface, ts::SplitRungeKutta3TimeStepper, timestepper, ::Val{3})

    η = free_surface.η
    U, V = free_surface.barotropic_velocities

    Uⁿ⁻¹ = ts.previous_model_fields.U
    Vⁿ⁻¹ = ts.previous_model_fields.v

    parent(U) .= parent(Uⁿ⁻¹)
    parent(V) .= parent(Vⁿ⁻¹)

    initialize_free_surface_timestepper!(timestepper, η, U, V)

    fill!(filtered_state.η, 0)
    fill!(filtered_state.U, 0)
    fill!(filtered_state.V, 0)

    return nothing
end

#####
##### Compute slow tendencies for the AB2 timestepper
#####

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


function split_explicit_forcing!(GUⁿ, GVⁿ, GU⁻, GV⁻, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, timestepper::QuasiAdamsBashforth2TimeStepper, stage)     active_cells_map = retrieve_surface_active_cells_map(grid)
    active_cells_map = retrieve_surface_active_cells_map(grid)
    launch!(architecture(grid), grid, :xy, _compute_integrated_ab2_tendencies!, GUⁿ, GVⁿ, grid,
            active_cells_map, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, timestepper.χ; active_cells_map)

    return nothing
end 

#####
##### Compute slow tendencies for the RK3 timestepper
#####

@inline function vertical_integral(i, j, grid, Gⁿ, ℓx, ℓy, ℓz)
    G = Δz(i, j, 1, grid, ℓx, ℓy, ℓz) * ifelse(peripheral_node(i, j, 1, grid, ℓx, ℓy, ℓz), zero(grid), Gⁿ[i, j, 1])
    
    for k in 2:grid.Nz	
        @inbounds G += Δz(i, j, k, grid, ℓx, ℓy, ℓz) * ifelse(peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz), zero(grid), Gⁿ[i, j, k])
    end
    
    return G
end

@kernel function _compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, grid, active_cells_map, Guⁿ, Gvⁿ, stage)
    idx = @index(Global, Linear)
    i, j = active_linear_index_to_tuple(idx, active_cells_map)
    compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, i, j, grid, Guⁿ, Gvⁿ, stage)
end

@kernel function _compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, grid, ::Nothing, Guⁿ, Gvⁿ, stage)
    i, j = @index(Global, NTuple)
    compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, i, j, grid, Guⁿ, Gvⁿ, stage)
end

function compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, i, j, grid, Guⁿ, Gvⁿ, ::Val{1})
    @inbounds GUⁿ[i, j, 1] = vertical_integral(i, j, grid, Guⁿ, Face(), Center(), Center())
    @inbounds GVⁿ[i, j, 1] = vertical_integral(i, j, grid, Gvⁿ, Center(), Face(), Center())

    @inbounds GU⁻[i, j, 1] = GUⁿ[i, j, 1]
    @inbounds GV⁻[i, j, 1] = GVⁿ[i, j, 1]
end

@kernel function _compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, grid, Guⁿ, Gvⁿ, ::Val{2})
    i, j = @index(Global, NTuple)

    FT = eltype(GUⁿ)

    @inbounds GUⁿ[i, j, 1] = vertical_integral(i, j, grid, Guⁿ, Face(), Center(), Center())
    @inbounds GVⁿ[i, j, 1] = vertical_integral(i, j, grid, Gvⁿ, Center(), Face(), Center())

    @inbounds GU⁻[i, j, 1] = convert(FT, 1/6) * GUⁿ[i, j, 1] + convert(FT, 1/6) * GU⁻[i, j, 1]
    @inbounds GV⁻[i, j, 1] = convert(FT, 1/6) * GVⁿ[i, j, 1] + convert(FT, 1/6) * GU⁻[i, j, 1]
end

@kernel function _compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, grid, Guⁿ, Gvⁿ, ::Val{3})
    i, j = @index(Global, NTuple)

    FT = eltype(GUⁿ)

    @inbounds GUⁿ[i, j, 1] = vertical_integral(i, j, grid, Guⁿ, Face(), Center(), Center())
    @inbounds GVⁿ[i, j, 1] = vertical_integral(i, j, grid, Gvⁿ, Center(), Face(), Center())

    @inbounds GUⁿ[i, j, 1] = convert(FT, 2/3) * GUⁿ[i, j, 1] + GU⁻[i, j, 1]
    @inbounds GVⁿ[i, j, 1] = convert(FT, 2/3) * GVⁿ[i, j, 1] + GV⁻[i, j, 1]
end

function split_explicit_forcing!(GUⁿ, GVⁿ, GU⁻, GV⁻, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, ::SplitRungeKutta3TimeStepper, stage)  
    active_cells_map = retrieve_surface_active_cells_map(grid)    
    launch!(architecture(grid), grid, :xy, _compute_integrated_rk3_tendencies!, 
            GUⁿ, GVⁿ, GU⁻, GV⁻, grid, active_cells_map, Guⁿ, Gvⁿ, Val(stage); active_cells_map)

    return nothing
end 

#####
##### Free surface setup
#####

# Setting up the RHS for the barotropic step (tendencies of the barotropic velocity components)
# This function is called after `calculate_tendency` and before `ab2_step_velocities!`
function setup_free_surface!(model, free_surface::SplitExplicitFreeSurface, baroclinic_timestepper, stage)
    
    # we start the time integration of η from the average ηⁿ     
    Gu⁻ = model.timestepper.G⁻.u
    Gv⁻ = model.timestepper.G⁻.v
    Guⁿ = model.timestepper.Gⁿ.u
    Gvⁿ = model.timestepper.Gⁿ.v

    GU⁻ = model.timestepper.G⁻.U
    GV⁻ = model.timestepper.G⁻.V
    GUⁿ = model.timestepper.Gⁿ.U
    GVⁿ = model.timestepper.Gⁿ.V

    barotropic_timestepper = free_surface.timestepper
    
    @apply_regionally split_explicit_forcing!(GUⁿ, GVⁿ, GU⁻, GV⁻, model.grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, baroclinic_timestepper, stage)

    initialize_free_surface_state!(free_surface, baroclinic_timestepper, barotropic_timestepper, Val(stage))

    fields_to_fill = (GUⁿ, GVⁿ)
    fill_halo_regions!(fields_to_fill; async = true)

    return nothing
end