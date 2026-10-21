using Oceananigans.Operators: Δx, Δy, Δz

#####
##### The halo beyond a Value, Gradient or Mixed wall holds the wall value, half the first interior cell from its node
#####

@inline linearly_extrapolate(c₀, ∇c, Δ) = c₀ + ∇c * Δ

@inline  left_gradient(bc::GBC, c¹, Δ, i, j, args...) = getbc(bc, i, j, args...)
@inline right_gradient(bc::GBC, cᴺ, Δ, i, j, args...) = getbc(bc, i, j, args...)

@inline  left_gradient(bc::VBC, c¹, Δ, i, j, args...) = ( c¹ - getbc(bc, i, j, args...) ) / (Δ/2)
@inline right_gradient(bc::VBC, cᴺ, Δ, i, j, args...) = ( getbc(bc, i, j, args...) - cᴺ ) / (Δ/2)

@inline function left_gradient(bc::MBC, c¹, Δ, i, j, args...)
    coefficient = bc.condition.coefficient
    inhomogeneity = bc.condition.inhomogeneity
    a = getbc(coefficient, i, j, args...)
    b = getbc(inhomogeneity, i, j, args...)
    g₁ = c¹ / Δ * (1 - (1 + a * Δ / 2) / (1 - a * Δ / 2))
    g₂ = b / (1 - a * Δ / 2)
    return g₁ + g₂
end

@inline function right_gradient(bc::MBC, cᴺ, Δ, i, j, args...)
    coefficient = bc.condition.coefficient
    inhomogeneity = bc.condition.inhomogeneity
    a = getbc(coefficient, i, j, args...)
    b = getbc(inhomogeneity, i, j, args...)
    g₁ = cᴺ / Δ * ((1 - a * Δ / 2) / (1 + a * Δ / 2) - 1)
    g₂ = b / (1 + a * Δ / 2)
    return g₁ + g₂
end

@inline function _fill_west_halo!(j, k, grid, c, bc::Union{VBC, GBC, MBC}, loc, args...)

           #  ↑ x ↑  interior
           #  -----  interior face
    iᴵ = 1 #    *    interior cell
           #  =====  western boundary
    iᴴ = 0 #    *    halo cell

    LX, LY, LZ = loc
    Δ = Δx(iᴵ, j, k, grid, LX, LY, LZ)
    @inbounds ∇c = left_gradient(bc, c[iᴵ, j, k], Δ, j, k, grid, args...)
    @inbounds c[iᴴ, j, k] = linearly_extrapolate(c[iᴵ, j, k], ∇c, -Δ / 2)
end

@inline function _fill_east_halo!(j, k, grid, c, bc::Union{VBC, GBC, MBC}, loc, args...)

                     #  ↑ x ↑
    iᴴ = grid.Nx + 1 #    *   halo cell
                     #  ===== eastern boundary
    iᴵ = grid.Nx     #    *   interior cell
                     #  ----- interior face
                     #    ↓   interior

    LX, LY, LZ = loc
    Δ = Δx(iᴵ, j, k, grid, LX, LY, LZ)
    @inbounds ∇c = right_gradient(bc, c[iᴵ, j, k], Δ, j, k, grid, args...)
    @inbounds c[iᴴ, j, k] = linearly_extrapolate(c[iᴵ, j, k], ∇c, Δ / 2)
end

@inline function _fill_south_halo!(i, k, grid, c, bc::Union{VBC, GBC, MBC}, loc, args...)

           #  ↑ y ↑  interior
           #  -----  interior face
    jᴵ = 1 #    *    interior cell
           #  =====  southern boundary
    jᴴ = 0 #    *    halo cell

    LX, LY, LZ = loc
    Δ = Δy(i, jᴵ, k, grid, LX, LY, LZ)
    @inbounds ∇c = left_gradient(bc, c[i, jᴵ, k], Δ, i, k, grid, args...)
    @inbounds c[i, jᴴ, k] = linearly_extrapolate(c[i, jᴵ, k], ∇c, -Δ / 2)
end

@inline function _fill_north_halo!(i, k, grid, c, bc::Union{VBC, GBC, MBC}, loc, args...)

                     #  ↑ y ↑
    jᴴ = grid.Ny + 1 #    *   halo cell
                     #  ===== northern boundary
    jᴵ = grid.Ny     #    *   interior cell
                     #  ----- interior face
                     #    ↓   interior

    LX, LY, LZ = loc
    Δ = Δy(i, jᴵ, k, grid, LX, LY, LZ)
    @inbounds ∇c = right_gradient(bc, c[i, jᴵ, k], Δ, i, k, grid, args...)
    @inbounds c[i, jᴴ, k] = linearly_extrapolate(c[i, jᴵ, k], ∇c, Δ / 2)
end

@inline function _fill_bottom_halo!(i, j, grid, c, bc::Union{VBC, GBC, MBC}, loc, args...)

           #  ↑ z ↑  interior
           #  -----  interior face
    kᴵ = 1 #    *    interior cell
           #  =====  bottom boundary
    kᴴ = 0 #    *    halo cell

    LX, LY, LZ = loc
    Δ = Δz(i, j, kᴵ, grid, LX, LY, LZ)
    @inbounds ∇c = left_gradient(bc, c[i, j, kᴵ], Δ, i, j, grid, args...)
    @inbounds c[i, j, kᴴ] = linearly_extrapolate(c[i, j, kᴵ], ∇c, -Δ / 2)
end

@inline function _fill_top_halo!(i, j, grid, c, bc::Union{VBC, GBC, MBC}, loc, args...)

                     #  ↑ z ↑
    kᴴ = grid.Nz + 1 #    *    halo cell
                     #  =====  top boundary
    kᴵ = grid.Nz     #    *    interior cell
                     #  -----  interior face

    LX, LY, LZ = loc
    Δ = Δz(i, j, kᴵ, grid, LX, LY, LZ)
    @inbounds ∇c = right_gradient(bc, c[i, j, kᴵ], Δ, i, j, grid, args...)
    @inbounds c[i, j, kᴴ] = linearly_extrapolate(c[i, j, kᴵ], ∇c, Δ / 2)
end
