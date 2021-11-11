function right_stencil_x(grid::RectilinearGrid{FT}, i, j, k, ψ) where FT
    u̅ = zeros(FT, 4)

    for h = -2:1
        γ =  Δxᶜᶜᵃ(i+h, j, k, grid)/(Δxᶜᶜᵃ(i+h, j, k, grid) + Δxᶜᶜᵃ(i+h+1, j, k, grid))
        u̅[h+3] = (1 - γ) * ψ[i+h, j, k] + γ * ψ[i+h+1, j, k]
    end

    # u̅⁺ = ((2 * Δzᵃᵃᶜ(i, j, k+1, grid) + Δzᵃᵃᶜ(i, j, k+2, grid)) * ψ[i, j, k+1] - Δzᵃᵃᶜ(i, j, k+1, grid) * ψ[i, j, k+2]) / 
    #           (Δzᵃᵃᶜ(i, j, k+1, grid) + Δzᵃᵃᶜ(i, j, k+2, grid))

    return ([u̅[1], ψ[i-1, j, k], u̅[2]],
            [u̅[2], ψ[i, j, k]  , u̅[3]],
            [u̅[3], ψ[i+1, j, k], u̅[4]])
            #[ψ[i, j, k+1], u̅[3], u̅⁺])
end

function left_stencil_x(grid::RectilinearGrid{FT}, i, j, k, ψ) where FT
    u̅ = zeros(FT, 4)

    for h = -3:0
        γ =  Δxᶜᶜᵃ(i + h, j, k, grid)/(Δxᶜᶜᵃ(i + h, j, k, grid) + Δxᶜᶜᵃ(i + h + 1, j, k, grid))
        u̅[h+4] = (1 - γ) * ψ[i + h, j, k] + γ * ψ[i + h + 1, j, k]
    end

    # u̅⁺ = ((2 * Δzᵃᵃᶜ(i, j, k-2, grid) + Δzᵃᵃᶜ(i, j, k-3, grid)) * ψ[i, j, k-2] - Δzᵃᵃᶜ(i, j, k-2, grid) * ψ[i, j, k-3]) / 
    #           (Δzᵃᵃᶜ(i, j, k-2, grid) + Δzᵃᵃᶜ(i, j, k-3, grid))

    #           #[u̅⁺  , u̅[1], ψ[i, j, k-2]]

    return ([u̅[1], ψ[i-2, j, k], u̅[2]],
            [u̅[2], ψ[i-1, j, k], u̅[3]],
            [u̅[3], ψ[i, j, k]  , u̅[4]])
end

function right_stencil_z(grid::RectilinearGrid{FT}, i, j, k, ψ) where FT
    u̅ = zeros(FT, 4)

    for h = -2:1
        γ =  Δzᵃᵃᶜ(i, j, k + h, grid)/(Δzᵃᵃᶜ(i, j, k + h, grid) + Δzᵃᵃᶜ(i, j, k + h + 1, grid))
        u̅[h+3] = (1 - γ) * ψ[i, j, k + h] + γ * ψ[i, j, k + h + 1]
    end

    # u̅⁺ = ((2 * Δzᵃᵃᶜ(i, j, k+1, grid) + Δzᵃᵃᶜ(i, j, k+2, grid)) * ψ[i, j, k+1] - Δzᵃᵃᶜ(i, j, k+1, grid) * ψ[i, j, k+2]) / 
    #           (Δzᵃᵃᶜ(i, j, k+1, grid) + Δzᵃᵃᶜ(i, j, k+2, grid))

    return ([u̅[1], ψ[i, j, k-1], u̅[2]],
            [u̅[2], ψ[i, j, k]  , u̅[3]],
            [u̅[3], ψ[i, j, k+1], u̅[4]])
            #[ψ[i, j, k+1], u̅[3], u̅⁺])
end

function left_stencil_z(grid::RectilinearGrid{FT}, i, j, k, ψ) where FT
    u̅ = zeros(FT, 4)

    for h = -3:0
        γ =  Δzᵃᵃᶜ(i, j, k + h, grid)/(Δzᵃᵃᶜ(i, j, k + h, grid) + Δzᵃᵃᶜ(i, j, k + h + 1, grid))
        u̅[h+4] = (1 - γ) * ψ[i, j, k + h] + γ * ψ[i, j, k + h + 1]
    end

    # u̅⁺ = ((2 * Δzᵃᵃᶜ(i, j, k-2, grid) + Δzᵃᵃᶜ(i, j, k-3, grid)) * ψ[i, j, k-2] - Δzᵃᵃᶜ(i, j, k-2, grid) * ψ[i, j, k-3]) / 
    #           (Δzᵃᵃᶜ(i, j, k-2, grid) + Δzᵃᵃᶜ(i, j, k-3, grid))

    #           #[u̅⁺  , u̅[1], ψ[i, j, k-2]]

    return ([u̅[1], ψ[i, j, k-2], u̅[2]],
            [u̅[2], ψ[i, j, k-1], u̅[3]],
            [u̅[3], ψ[i, j, k]  , u̅[4]])
end


# @inline left_biased_β₂(grid::AG{FT}, ψ) where FT = @inbounds FT(13/12)   * (2ψ[2] - 2ψ[1])^two_32 + FT(1/4) * ( - 2ψ[1] - 2ψ[2] + 4ψ[3])^two_32

# @inline right_biased_β₀(grid::AG{FT}, ψ) where FT = @inbounds FT(13/12) * (2ψ[2] - 2ψ[3])^two_32  + FT(1/4) * ( - 2ψ[3] - 2ψ[2] + 4ψ[1])^two_32
