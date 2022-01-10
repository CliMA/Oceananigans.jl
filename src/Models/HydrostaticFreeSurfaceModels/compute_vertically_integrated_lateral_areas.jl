
# Has to be changed when the regression data is updated 

@kernel function _compute_vertically_integrated_lateral_areas!(∫ᶻ_A, grid)
    i, j = @index(Global, NTuple)

    # to account for halos (which are (2, 2) for the implicit free surface)
    i, j .-= 2

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
    # a periodic domain, where it is not guaranteed that η₁ == ηₙ 
    # 2 halos are necessary to accomodate the preconditioner
    xy_size = size(grid)[[1, 2]] .+ 4
    
    event = launch!(arch, grid, xy_size,
                    _compute_vertically_integrated_lateral_areas!,
                    ∫ᶻ_A, grid,
                    dependencies=Event(device(arch)))

    wait(device(arch), event)

    return nothing
end
