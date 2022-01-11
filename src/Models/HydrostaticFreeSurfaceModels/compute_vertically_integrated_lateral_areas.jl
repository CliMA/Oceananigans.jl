
# Has to be changed when the regression data is updated 

@kernel function _compute_vertically_integrated_lateral_areas!(∫ᶻ_A, grid)
    i, j = @index(Global, NTuple)

    # to account for halos (Hx, Hy = 2, 2 for ∫ᶻ_A)
    # Hx, Hy, Hz = halo_size(grid)
    # i -= Hx
    # j -= Hy

    @inbounds begin
        ∫ᶻ_A.xᶠᶜᶜ[i, j, 1] = 0
        ∫ᶻ_A.yᶜᶠᶜ[i, j, 1] = 0

        @unroll for k in 1:grid.Nz
            ∫ᶻ_A.xᶠᶜᶜ[i, j, 1] += Δyᶠᶜᵃ(i, j, k, grid) * Δzᶠᶜᶜ(i, j, k, grid)
            ∫ᶻ_A.yᶜᶠᶜ[i, j, 1] += Δxᶜᶠᵃ(i, j, k, grid) * Δzᶜᶠᶜ(i, j, k, grid)
        end
    end
end

function compute_vertically_integrated_lateral_areas!(∫ᶻ_A, grid, arch)

    # we have to account for halos when calculating Integrated areas, in case 
    # a periodic domain, where it is not guaranteed that ηₙ == ηₙ₊₁ 
    # 2 halos (instead of only 1) are necessary to accomodate the preconditioner

    field_grid = ∫ᶻ_A.xᶠᶜᶜ.grid

    xy_size = size(field_grid)[[1, 2]] .+ halo_size(field_grid)[[1, 2]] .* 2
    
    event = launch!(arch, field_grid, :xy,
                    _compute_vertically_integrated_lateral_areas!,
                    ∫ᶻ_A, grid,
                    dependencies=Event(device(arch)))

    wait(device(arch), event)

    return nothing
end
