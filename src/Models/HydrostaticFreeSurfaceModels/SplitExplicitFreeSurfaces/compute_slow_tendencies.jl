
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

    # multiply G⁻ by false if C₂ is zero to 
    # prevent propagationg possible NaNs
    euler = C₂ != 0

    Gⁿ⁺¹ = @inbounds C₁ * Gⁿ[i, j, k] - C₂ * G⁻[i, j, k] * euler
    immersed = peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz)

    return ifelse(immersed, zero(grid), Gⁿ⁺¹)
end

# Setting up the RHS for the barotropic step (tendencies of the barotropic velocity components)
# This function is called after `calculate_tendency` and before `ab2_step_velocities!`
function compute_free_surface_tendency!(grid, model, ::SplitExplicitFreeSurface)

    @show "Computing tendency"
    # we start the time integration of η from the average ηⁿ
    Gu⁻ = model.timestepper.G⁻.u
    Gv⁻ = model.timestepper.G⁻.v
    Guⁿ = model.timestepper.Gⁿ.u
    Gvⁿ = model.timestepper.Gⁿ.v

    GUⁿ = model.timestepper.Gⁿ.U
    GVⁿ = model.timestepper.Gⁿ.V

    @apply_regionally compute_free_surface_forcing!(GUⁿ, GVⁿ, model.grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, model.timestepper.χ)

    fields_to_fill = (GUⁿ, GVⁿ)
    fill_halo_regions!(fields_to_fill; async = true)

    return nothing
end

@inline function compute_free_surface_forcing!(GUⁿ, GVⁿ, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)
    active_cells_map = retrieve_surface_active_cells_map(grid)

    launch!(architecture(grid), grid, :xy, _compute_integrated_ab2_tendencies!, GUⁿ, GVⁿ, grid,
            active_cells_map, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ; active_cells_map)

    return nothing
end