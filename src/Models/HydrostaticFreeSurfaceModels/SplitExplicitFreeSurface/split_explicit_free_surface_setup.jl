
# Calculate RHS for the barotropic time step.
@kernel function _compute_integrated_ab2_tendencies!(Gᵁ, Gⱽ, grid, ::Nothing, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)
    i, j  = @index(Global, NTuple)
    k_top = grid.Nz + 1

    @inbounds Gᵁ[i, j, k_top-1] = Δzᶠᶜᶜ(i, j, 1, grid) * ab2_step_Gu(i, j, 1, grid, Gu⁻, Guⁿ, χ)
    @inbounds Gⱽ[i, j, k_top-1] = Δzᶜᶠᶜ(i, j, 1, grid) * ab2_step_Gv(i, j, 1, grid, Gv⁻, Gvⁿ, χ)

    for k in 2:grid.Nz	
        @inbounds Gᵁ[i, j, k_top-1] += Δzᶠᶜᶜ(i, j, k, grid) * ab2_step_Gu(i, j, k, grid, Gu⁻, Guⁿ, χ)
        @inbounds Gⱽ[i, j, k_top-1] += Δzᶜᶠᶜ(i, j, k, grid) * ab2_step_Gv(i, j, k, grid, Gv⁻, Gvⁿ, χ)
    end	
end

# Calculate RHS for the barotropic time step.q
@kernel function _compute_integrated_ab2_tendencies!(Gᵁ, Gⱽ, grid, active_cells_map, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)
    idx = @index(Global, Linear)
    i, j = active_linear_index_to_tuple(idx, active_cells_map)
    k_top = grid.Nz+1

    @inbounds Gᵁ[i, j, k_top-1] = Δzᶠᶜᶜ(i, j, 1, grid) * ab2_step_Gu(i, j, 1, grid, Gu⁻, Guⁿ, χ)
    @inbounds Gⱽ[i, j, k_top-1] = Δzᶜᶠᶜ(i, j, 1, grid) * ab2_step_Gv(i, j, 1, grid, Gv⁻, Gvⁿ, χ)

    for k in 2:grid.Nz	
        @inbounds Gᵁ[i, j, k_top-1] += Δzᶠᶜᶜ(i, j, k, grid) * ab2_step_Gu(i, j, k, grid, Gu⁻, Guⁿ, χ)
        @inbounds Gⱽ[i, j, k_top-1] += Δzᶜᶠᶜ(i, j, k, grid) * ab2_step_Gv(i, j, k, grid, Gv⁻, Gvⁿ, χ)
    end	
end

@inline ab2_step_Gu(i, j, k, grid, G⁻, Gⁿ, χ::FT) where FT =
    @inbounds ifelse(peripheral_node(i, j, k, grid, f, c, c), zero(grid), (convert(FT, 1.5) + χ) *  Gⁿ[i, j, k] - G⁻[i, j, k] * (convert(FT, 0.5) + χ))

@inline ab2_step_Gv(i, j, k, grid, G⁻, Gⁿ, χ::FT) where FT =
    @inbounds ifelse(peripheral_node(i, j, k, grid, c, f, c), zero(grid), (convert(FT, 1.5) + χ) *  Gⁿ[i, j, k] - G⁻[i, j, k] * (convert(FT, 0.5) + χ))


@inline function ab2_split_explicit_forcing!(auxiliary, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ) 
    active_cells_map = retrieve_surface_active_cells_map(grid)

    launch!(architecture(grid), grid, :xy, _compute_integrated_ab2_tendencies!, auxiliary.Gᵁ, auxiliary.Gⱽ, grid, 
            active_cells_map, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ; active_cells_map)

    return nothing
end
            
# Setting up the RHS for the barotropic step (tendencies of the barotropic velocity components)
# This function is called after `calculate_tendency` and before `ab2_step_velocities!`
function setup_free_surface!(model, free_surface::SplitExplicitFreeSurface, timestepper::QuasiAdamsBashforth2TimeStepper, stage)
    
    χ = timestepper.χ

    # we start the time integration of η from the average ηⁿ     
    Gu⁻ = model.timestepper.G⁻.u
    Gv⁻ = model.timestepper.G⁻.v
    Guⁿ = model.timestepper.Gⁿ.u
    Gvⁿ = model.timestepper.Gⁿ.v

    auxiliary = free_surface.auxiliary

    @apply_regionally ab2_split_explicit_forcing!(auxiliary, model.grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)

    fields_to_fill = (auxiliary.Gᵁ, auxiliary.Gⱽ)
    fill_halo_regions!(fields_to_fill; async = true)

    return nothing
end

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

function ssprk3_split_explicit_forcing!(auxiliary, grid, Guⁿ, Gvⁿ, ::Val{1}) 
    Gᵁⁿ = auxiliary.Gᵁ.new
    Gⱽⁿ = auxiliary.Gⱽ.new

    launch!(architecture(grid), grid, :xy, _compute_integrated_tendencies!, Gᵁⁿ, Gⱽⁿ, grid, Guⁿ, Gvⁿ)

    return nothing
end

function ssprk3_split_explicit_forcing!(auxiliary, grid, Guⁿ, Gvⁿ, ::Val{2}) 
    Gᵁⁿ = auxiliary.Gᵁ.new
    Gⱽⁿ = auxiliary.Gⱽ.new
    Gᵁᵒ = auxiliary.Gᵁ.old
    Gⱽᵒ = auxiliary.Gⱽ.old

    parent(Gᵁᵒ) .= parent(Gᵁⁿ)
    parent(Gⱽᵒ) .= parent(Gⱽⁿ)

    launch!(architecture(grid), grid, :xy, _compute_integrated_tendencies!, Gᵁⁿ, Gⱽⁿ, grid, Guⁿ, Gvⁿ)

    return nothing
end

function ssprk3_split_explicit_forcing!(auxiliary, grid, Guⁿ, Gvⁿ, ::Val{3}) 
    Gᵁⁿ = auxiliary.Gᵁ.new
    Gⱽⁿ = auxiliary.Gⱽ.new
    Gᵁᵒ = auxiliary.Gᵁ.old
    Gⱽᵒ = auxiliary.Gⱽ.old

    parent(Gᵁᵒ) .= 1 // 6 .* parent(Gᵁⁿ) .+ 1 // 6 .* parent(Gᵁᵒ)
    parent(Gⱽᵒ) .= 1 // 6 .* parent(Gⱽⁿ) .+ 1 // 6 .* parent(Gⱽᵒ)

    launch!(architecture(grid), grid, :xy, _compute_integrated_tendencies!, Gᵁⁿ, Gⱽⁿ, grid, Guⁿ, Gvⁿ)

    parent(Gᵁⁿ) .= parent(Gᵁᵒ) .+ 2 // 3 .* parent(Gᵁⁿ)
    parent(Gⱽⁿ) .= parent(Gⱽᵒ) .+ 2 // 3 .* parent(Gⱽⁿ)

    return nothing
end

function setup_free_surface!(model, free_surface::SplitExplicitFreeSurface, ::SSPRK3TimeStepper, stage)

    # we start the time integration of η from the average ηⁿ     
    Guⁿ = model.timestepper.Gⁿ.u
    Gvⁿ = model.timestepper.Gⁿ.v

    auxiliary = free_surface.auxiliary

    @apply_regionally ssprk3_split_explicit_forcing!(auxiliary, model.grid, Guⁿ, Gvⁿ, Val(stage))

    fields_to_fill = (auxiliary.Gᵁ, auxiliary.Gⱽ)
    fill_halo_regions!(fields_to_fill; async = true)

    return nothing
end

wait_free_surface_communication!(free_surface, arch) = nothing

