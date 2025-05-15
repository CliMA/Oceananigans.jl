#####
##### Compute slow tendencies with an AB2 timestepper
#####

# Calculate RHS for the barotropic time step.
@kernel function _compute_integrated_ab2_tendencies!(Gᵁ, Gⱽ, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)
    i, j  = @index(Global, NTuple)

    locU = (Face(), Center(), Center())
    locV = (Center(), Face(), Center())

    @inbounds Gᵁ[i, j, 1] = Δzᶠᶜᶜ(i, j, 1, grid) * ab2_step_G(i, j, 1, grid, locU..., Gu⁻, Guⁿ, χ)
    @inbounds Gⱽ[i, j, 1] = Δzᶜᶠᶜ(i, j, 1, grid) * ab2_step_G(i, j, 1, grid, locV..., Gv⁻, Gvⁿ, χ)

    for k in 2:grid.Nz
        @inbounds Gᵁ[i, j, 1] += Δzᶠᶜᶜ(i, j, k, grid) * ab2_step_G(i, j, k, grid, locU..., Gu⁻, Guⁿ, χ)
        @inbounds Gⱽ[i, j, 1] += Δzᶜᶠᶜ(i, j, k, grid) * ab2_step_G(i, j, k, grid, locV..., Gv⁻, Gvⁿ, χ)
    end
end

@inline function ab2_step_G(i, j, k, grid, ℓx, ℓy, ℓz, G⁻, Gⁿ, χ)
    C₁ = 3 * one(grid) / 2 + χ
    C₂ =     one(grid) / 2 + χ

    # multiply G⁻ by false if C₂ is zero to
    # prevent propagationg possible NaNs
    not_euler = C₂ != 0

    Gⁿ⁺¹ = @inbounds C₁ * Gⁿ[i, j, k] - C₂ * G⁻[i, j, k] * not_euler
    immersed = peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz)

    return ifelse(immersed, zero(grid), Gⁿ⁺¹)
end

@inline function compute_split_explicit_forcing!(GUⁿ, GVⁿ, grid, Guⁿ, Gvⁿ,
                                                 timestepper::QuasiAdamsBashforth2TimeStepper, stage)
    active_cells_map = get_active_column_map(grid)

    Gu⁻ = timestepper.G⁻.u
    Gv⁻ = timestepper.G⁻.v

    launch!(architecture(grid), grid, :xy, _compute_integrated_ab2_tendencies!, GUⁿ, GVⁿ, grid,
            Gu⁻, Gv⁻, Guⁿ, Gvⁿ, timestepper.χ; active_cells_map)

    return nothing
end

#####
##### Compute slow tendencies with a RK3 timestepper
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

@kernel function _compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, grid, Guⁿ, Gvⁿ, stage)
    i, j = @index(Global, NTuple)
    compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, i, j, grid, Guⁿ, Gvⁿ, stage)
end

@inline function compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, i, j, grid, Guⁿ, Gvⁿ, ::Val{1})
    @inbounds GUⁿ[i, j, 1] = G_vertical_integral(i, j, grid, Guⁿ, Face(), Center(), Center())
    @inbounds GVⁿ[i, j, 1] = G_vertical_integral(i, j, grid, Gvⁿ, Center(), Face(), Center())

    @inbounds GU⁻[i, j, 1] = GUⁿ[i, j, 1]
    @inbounds GV⁻[i, j, 1] = GVⁿ[i, j, 1]

    return nothing
end

@inline function compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, i, j, grid, Guⁿ, Gvⁿ, ::Val{2})
    @inbounds GUⁿ[i, j, 1] = G_vertical_integral(i, j, grid, Guⁿ, Face(), Center(), Center())
    @inbounds GVⁿ[i, j, 1] = G_vertical_integral(i, j, grid, Gvⁿ, Center(), Face(), Center())

    @inbounds GU⁻[i, j, 1] = (GUⁿ[i, j, 1] + GU⁻[i, j, 1]) / 6
    @inbounds GV⁻[i, j, 1] = (GVⁿ[i, j, 1] + GV⁻[i, j, 1]) / 6

    return nothing
end

@inline function compute_integrated_rk3_tendencies!(GUⁿ, GVⁿ, GU⁻, GV⁻, i, j, grid, Guⁿ, Gvⁿ, ::Val{3})
    GUi = G_vertical_integral(i, j, grid, Guⁿ, Face(), Center(), Center())
    GVi = G_vertical_integral(i, j, grid, Gvⁿ, Center(), Face(), Center())

    @inbounds GUⁿ[i, j, 1] = 2 * GUi / 3 + GU⁻[i, j, 1]
    @inbounds GVⁿ[i, j, 1] = 2 * GVi / 3 + GV⁻[i, j, 1]

    return nothing
end

@inline function compute_split_explicit_forcing!(GUⁿ, GVⁿ, grid, Guⁿ, Gvⁿ,
                                                 timestepper::SplitRungeKutta3TimeStepper, stage)

    GU⁻ = timestepper.G⁻.U
    GV⁻ = timestepper.G⁻.V

    active_cells_map = get_active_column_map(grid)    
    launch!(architecture(grid), grid, :xy, _compute_integrated_rk3_tendencies!, 
            GUⁿ, GVⁿ, GU⁻, GV⁻, grid, Guⁿ, Gvⁿ, stage; active_cells_map)

    return nothing
end

#####
##### Free surface setup
#####

# Setting up the RHS for the barotropic step (tendencies of the barotropic velocity components)
# This function is called after `calculate_tendency` and before `ab2_step_velocities!`
function compute_free_surface_tendency!(grid, model, free_surface::SplitExplicitFreeSurface)

    Guⁿ = model.timestepper.Gⁿ.u
    Gvⁿ = model.timestepper.Gⁿ.v

    GUⁿ = model.timestepper.Gⁿ.U
    GVⁿ = model.timestepper.Gⁿ.V

    barotropic_timestepper = free_surface.timestepper
    baroclinic_timestepper = model.timestepper

    stage = model.clock.stage

    @apply_regionally begin
        compute_split_explicit_forcing!(GUⁿ, GVⁿ, grid, Guⁿ, Gvⁿ, baroclinic_timestepper, Val(stage))
        initialize_free_surface_state!(free_surface, baroclinic_timestepper, barotropic_timestepper, Val(stage))
    end

    fields_to_fill = (GUⁿ, GVⁿ)
    fill_halo_regions!(fields_to_fill; async = true)

    return nothing
end
