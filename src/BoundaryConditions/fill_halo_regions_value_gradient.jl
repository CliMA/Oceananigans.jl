using Oceananigans.Operators: Δx, Δy, Δz

#####
##### Halo filling for value and gradient boundary conditions
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
    iᴮ = 1 #  =====  western boundary
           #    *    halo cells (1 to Hx deep)

    LX, LY, LZ = loc
    Δ = Δx(iᴮ, j, k, grid, flip(LX), LY, LZ) # Δ between first interior and first west halo point, defined at cell face.
    @inbounds ∇c = left_gradient(bc, c[iᴵ, j, k], Δ, j, k, grid, args...)
    @inbounds for h in 1:grid.Hx
        c[1 - h, j, k] = linearly_extrapolate(c[iᴵ, j, k], ∇c, -h * Δ) # extrapolate westward in -x direction.
    end
end

@inline function _fill_east_halo!(j, k, grid, c, bc::Union{VBC, GBC, MBC}, loc, args...)

                     #  ↑ x ↑
                     #    *   halo cells (1 to Hx deep)
    iᴮ = grid.Nx + 1 #  ===== eastern boundary
    iᴵ = grid.Nx     #    *   interior cell
                     #  ----- interior face
                     #    ↓   interior

    LX, LY, LZ = loc
    Δ = Δx(iᴮ, j, k, grid, flip(LX), LY, LZ) # Δ between last interior and first east halo point, defined at cell face.
    @inbounds ∇c = right_gradient(bc, c[iᴵ, j, k], Δ, j, k, grid, args...)
    @inbounds for h in 1:grid.Hx
        c[grid.Nx + h, j, k] = linearly_extrapolate(c[iᴵ, j, k], ∇c, h * Δ) # extrapolate eastward in +x direction.
    end
end

@inline function _fill_south_halo!(i, k, grid, c, bc::Union{VBC, GBC, MBC}, loc, args...)

           #  ↑ y ↑  interior
           #  -----  interior face
    jᴵ = 1 #    *    interior cell
    jᴮ = 1 #  =====  southern boundary
           #    *    halo cells (1 to Hy deep)

    LX, LY, LZ = loc
    Δ = Δy(i, jᴮ, k, grid, LX, flip(LY), LZ) # Δ between first interior and first south halo point, defined at cell face.
    @inbounds ∇c = left_gradient(bc, c[i, jᴵ, k], Δ, i, k, grid, args...)
    @inbounds for h in 1:grid.Hy
        c[i, 1 - h, k] = linearly_extrapolate(c[i, jᴵ, k], ∇c, -h * Δ) # extrapolate southward in -y direction.
    end
end

@inline function _fill_north_halo!(i, k, grid, c, bc::Union{VBC, GBC, MBC}, loc, args...)

                     #  ↑ y ↑
                     #    *   halo cells (1 to Hy deep)
    jᴮ = grid.Ny + 1 #  ===== northern boundary
    jᴵ = grid.Ny     #    *   interior cell
                     #  ----- interior face
                     #    ↓   interior

    LX, LY, LZ = loc
    Δ = Δy(i, jᴮ, k, grid, LX, flip(LY), LZ) # Δ between first interior and first north halo point, defined at cell face.
    @inbounds ∇c = right_gradient(bc, c[i, jᴵ, k], Δ, i, k, grid, args...)
    @inbounds for h in 1:grid.Hy
        c[i, grid.Ny + h, k] = linearly_extrapolate(c[i, jᴵ, k], ∇c, h * Δ) # extrapolate northward in +y direction.
    end
end

@inline function _fill_bottom_halo!(i, j, grid, c, bc::Union{VBC, GBC, MBC}, loc, args...)

           #  ↑ z ↑  interior
           #  -----  interior face
    kᴵ = 1 #    *    interior cell
    kᴮ = 1 #  =====  bottom boundary
           #    *    halo cells (1 to Hz deep)

    LX, LY, LZ = loc
    Δ = Δz(i, j, kᴮ, grid, LX, LY, flip(LZ)) # Δ between first interior and first bottom halo point, defined at cell face.
    @inbounds ∇c = left_gradient(bc, c[i, j, kᴵ], Δ, i, j, grid, args...)
    @inbounds for h in 1:grid.Hz
        c[i, j, 1 - h] = linearly_extrapolate(c[i, j, kᴵ], ∇c, -h * Δ) # extrapolate downward in -z direction.
    end
end

@inline function _fill_top_halo!(i, j, grid, c, bc::Union{VBC, GBC, MBC}, loc, args...)

                     #  ↑ z ↑
                     #    *    halo cells (1 to Hz deep)
    kᴮ = grid.Nz + 1 #  =====  top boundary
    kᴵ = grid.Nz     #    *    interior cell
                     #  -----  interior face

    LX, LY, LZ = loc
    Δ = Δz(i, j, kᴮ, grid, LX, LY, flip(LZ)) # Δ between first interior and first top halo point, defined at cell face.
    @inbounds ∇c = right_gradient(bc, c[i, j, kᴵ], Δ, i, j, grid, args...)
    @inbounds for h in 1:grid.Hz
        c[i, j, grid.Nz + h] = linearly_extrapolate(c[i, j, kᴵ], ∇c, h * Δ) # extrapolate upward in +z direction.
    end
end
