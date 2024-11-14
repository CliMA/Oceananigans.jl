#####
##### Compute slow tendencies with an AB2 timestepper
#####

# Calculate RHS for the barotropic time step.
@kernel function _compute_integrated_ab2_tendencies!(Gᵁ, Gⱽ, grid, ::Nothing, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)
    i, j  = @index(Global, NTuple)
    ab2_integrate_tendencies!(Gᵁ, Gⱽ, i, j, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)
end

@kernel function _compute_integrated_ab2_tendencies!(Gᵁ, Gⱽ, grid, active_cells_map, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)
    idx = @index(Global, Linear)
    i, j = active_linear_index_to_tuple(idx, active_cells_map)
    ab2_integrate_tendencies!(Gᵁ, Gⱽ, i, j, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)
end

@inline function ab2_integrate_tendencies!(Gᵁ, Gⱽ, i, j, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)
    locU = (Face(), Center(), Center())
    locV = (Center(), Face(), Center())

    @inbounds Gᵁ[i, j, 1] = Δzᶠᶜᶜ(i, j, 1, grid) * ab2_step_G(i, j, 1, grid, locU..., Gu⁻, Guⁿ, χ)
    @inbounds Gⱽ[i, j, 1] = Δzᶜᶠᶜ(i, j, 1, grid) * ab2_step_G(i, j, 1, grid, locV..., Gv⁻, Gvⁿ, χ)

    for k in 2:grid.Nz
        @inbounds Gᵁ[i, j, 1] += Δzᶠᶜᶜ(i, j, k, grid) * ab2_step_G(i, j, k, grid, locU..., Gu⁻, Guⁿ, χ)
        @inbounds Gⱽ[i, j, 1] += Δzᶜᶠᶜ(i, j, k, grid) * ab2_step_G(i, j, k, grid, locV..., Gv⁻, Gvⁿ, χ)
    end
end

@inline function ab2_step_G(i, j, k, grid, ℓx, ℓy, ℓz, G⁻, Gⁿ, χ::FT) where FT 
    C₁ = convert(FT, 3/2) + χ
    C₂ = convert(FT, 1/2) + χ
    
    Gⁿ⁺¹ = @inbounds C₁ * Gⁿ[i, j, k] - C₂ * G⁻[i, j, k]
    immersed = peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz)

    return ifelse(immersed, zero(grid), Gⁿ⁺¹)
end

@inline function compute_split_explicit_forcing!(GUⁿ, GVⁿ, GU⁻, GV⁻, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, 
                                                 timestepper::QuasiAdamsBashforth2TimeStepper, stage)
    active_cells_map = retrieve_surface_active_cells_map(grid)

    launch!(architecture(grid), grid, :xy, _compute_integrated_ab2_tendencies!, GUⁿ, GVⁿ, grid,
            active_cells_map, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, timestepper.χ; active_cells_map)

    return nothing
end


#####
##### Compute slow tendencies with an RK3 timestepper
#####

@inline function G_vertical_integral(i, j, grid, Gⁿ, ℓx, ℓy, ℓz)
    immersed = peripheral_node(i, j, 1, grid, ℓx, ℓy, ℓz)

    Gⁿ⁺¹ = Δz(i, j, 1, grid, ℓx, ℓy, ℓz) * ifelse(immersed, zero(grid), Gⁿ[i, j, 1])
    
    @inbounds for k in 2:grid.Nz	
        immersed = peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz)
        Gⁿ⁺¹    += Δz(i, j, k, grid, ℓx, ℓy, ℓz) * ifelse(immersed, zero(grid), Gⁿ[i, j, k])
    end
    
    return Gⁿ⁺¹
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

@inline function compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, i, j, grid, Guⁿ, Gvⁿ, ::Val{1})
    @inbounds GUⁿ[i, j, 1] = G_vertical_integral(i, j, grid, Guⁿ, Face(), Center(), Center())
    @inbounds GVⁿ[i, j, 1] = G_vertical_integral(i, j, grid, Gvⁿ, Center(), Face(), Center())

    @inbounds GU⁻[i, j, 1] = GUⁿ[i, j, 1]
    @inbounds GV⁻[i, j, 1] = GVⁿ[i, j, 1]
end

@inline function compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, i, j, grid, Guⁿ, Gvⁿ, ::Val{2})
    FT = eltype(GUⁿ)

    @inbounds GUⁿ[i, j, 1] = G_vertical_integral(i, j, grid, Guⁿ, Face(), Center(), Center())
    @inbounds GVⁿ[i, j, 1] = G_vertical_integral(i, j, grid, Gvⁿ, Center(), Face(), Center())

    @inbounds GU⁻[i, j, 1] = convert(FT, 1/6) * GUⁿ[i, j, 1] + convert(FT, 1/6) * GU⁻[i, j, 1]
    @inbounds GV⁻[i, j, 1] = convert(FT, 1/6) * GVⁿ[i, j, 1] + convert(FT, 1/6) * GU⁻[i, j, 1]
end

@inline function compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, i, j, grid, Guⁿ, Gvⁿ, ::Val{3})
    FT = eltype(GUⁿ)

    @inbounds GUⁿ[i, j, 1] = G_vertical_integral(i, j, grid, Guⁿ, Face(), Center(), Center())
    @inbounds GVⁿ[i, j, 1] = G_vertical_integral(i, j, grid, Gvⁿ, Center(), Face(), Center())

    @inbounds GUⁿ[i, j, 1] = convert(FT, 2/3) * GUⁿ[i, j, 1] + GU⁻[i, j, 1]
    @inbounds GVⁿ[i, j, 1] = convert(FT, 2/3) * GVⁿ[i, j, 1] + GV⁻[i, j, 1]
end

@inline function compute_split_explicit_forcing!(GUⁿ, GVⁿ, GU⁻, GV⁻, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, 
                                                 ::SplitRungeKutta3TimeStepper, stage)

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
function compute_free_surface_tendency!(grid, model, free_surface::SplitExplicitFreeSurface)

    G⁻ = model.timestepper.G⁻

    # we start the time integration of η from the average ηⁿ
    Gu⁻ = haskey(G⁻, :u) ? G⁻.u : nothing
    Gv⁻ = haskey(G⁻, :v) ? G⁻.v : nothing
    Guⁿ = model.timestepper.Gⁿ.u
    Gvⁿ = model.timestepper.Gⁿ.v

    GU⁻ = model.timestepper.G⁻.U
    GV⁻ = model.timestepper.G⁻.V
    GUⁿ = model.timestepper.Gⁿ.U
    GVⁿ = model.timestepper.Gⁿ.V

    barotropic_timestepper = free_surface.timestepper
    baroclinic_timestepper = model.timestepper
    stage = model.clock.stage

    @apply_regionally begin
        compute_split_explicit_forcing!(GUⁿ, GVⁿ, GU⁻, GV⁻, model.grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, baroclinic_timestepper, stage)
        initialize_free_surface_state!(free_surface, baroclinic_timestepper, barotropic_timestepper, Val(stage))
    end

    fields_to_fill = (GUⁿ, GVⁿ)
    fill_halo_regions!(fields_to_fill; async = true)

    return nothing
end
