using Oceananigans.Grids: AbstractUnderlyingGrid,
                          Bounded,
                          LeftConnected,
                          Periodic,
                          RightConnected,
                          RightCenterFolded,
                          RightFaceFolded

const AGXB  = AbstractUnderlyingGrid{FT, Bounded} where FT
const AGXP  = AbstractUnderlyingGrid{FT, Periodic} where FT
const AGXR  = AbstractUnderlyingGrid{FT, RightConnected} where FT
const AGXL  = AbstractUnderlyingGrid{FT, LeftConnected} where FT

const AGYB  = AbstractUnderlyingGrid{FT, <:Any, Bounded} where FT
const AGYP  = AbstractUnderlyingGrid{FT, <:Any, Periodic} where FT
const AGYR  = AbstractUnderlyingGrid{FT, <:Any, <:Union{RightConnected, RightCenterFolded}} where FT
const AGYL  = AbstractUnderlyingGrid{FT, <:Any, LeftConnected} where FT
const AGYCF = AbstractUnderlyingGrid{FT, <:Any, RightCenterFolded} where FT
const AGYFF = AbstractUnderlyingGrid{FT, <:Any, RightFaceFolded} where FT

# Topology-aware Operators with the following convention:
#
#   `δxTᶠᵃᵃ` : Hardcodes `Noflux` or `Periodic` boundary conditions for a (Center, Center, Center) function `f` in the x-direction.
#   `δyTᵃᶠᵃ` : Hardcodes `Noflux` or `Periodic` boundary conditions for a (Center, Center, Center) function `f` in the y-direction
#
#   `δxTᶜᵃᵃ` : Hardcodes `NoPenetration` or `Periodic` boundary conditions for a (Face, Center, Center) function `U` in x direction
#   `δyTᵃᶜᵃ` : Hardcodes `NoPenetration` or `Periodic` boundary conditions for a (Center, Face, Center) function `V` in y direction
#
# Note: The naming convention is that `T` denotes a topology-aware operator. So `δxTᶠᵃᵃ` is the topology-aware version of `δxᶠᵃᵃ`.

# Fallback
@inline δxTᶠᵃᵃ(i, j, k, grid, f, args...) = δxᶠᵃᵃ(i, j, k, grid, f, args...)
@inline δyTᵃᶠᵃ(i, j, k, grid, f, args...) = δyᵃᶠᵃ(i, j, k, grid, f, args...)
@inline δxTᶜᵃᵃ(i, j, k, grid, f, args...) = δxᶜᵃᵃ(i, j, k, grid, f, args...)
@inline δyTᵃᶜᵃ(i, j, k, grid, f, args...) = δyᵃᶜᵃ(i, j, k, grid, f, args...)

# Enforce Periodic conditions
@inline δxTᶠᵃᵃ(i, j, k, grid::AGXP, f, args...) = f(i, j, k, grid, args...) - f(ifelse(i == 1, grid.Nx, i - 1), j, k, grid, args...)
@inline δyTᵃᶠᵃ(i, j, k, grid::AGYP, f, args...) = f(i, j, k, grid, args...) - f(i, ifelse(j == 1, grid.Ny, j - 1), k, grid, args...)

@inline δxTᶠᵃᵃ(i, j, k, grid::AGXP, c::AbstractArray) = @inbounds c[i, j, k] - c[ifelse(i == 1, grid.Nx, i - 1), j, k]
@inline δyTᵃᶠᵃ(i, j, k, grid::AGYP, c::AbstractArray) = @inbounds c[i, j, k] - c[i, ifelse(j == 1, grid.Ny, j - 1), k]

@inline δxTᶜᵃᵃ(i, j, k, grid::AGXP, f, args...) = f(ifelse(i == grid.Nx, 1, i + 1), j, k, grid, args...) - f(i, j, k, grid, args...)
@inline δyTᵃᶜᵃ(i, j, k, grid::AGYP, f, args...) = f(i, ifelse(j == grid.Ny, 1, j + 1), k, grid, args...) - f(i, j, k, grid, args...)

@inline δxTᶜᵃᵃ(i, j, k, grid::AGXP, u::AbstractArray) = @inbounds u[ifelse(i == grid.Nx, 1, i + 1), j, k] - u[i, j, k]
@inline δyTᵃᶜᵃ(i, j, k, grid::AGYP, v::AbstractArray) = @inbounds v[i, ifelse(j == grid.Ny, 1, j + 1), k] - v[i, j, k]

# Enforce NoFlux conditions
@inline δxTᶠᵃᵃ(i, j, k, grid::AGXB{FT}, f, args...) where FT = ifelse(i == 1, zero(FT), δxᶠᵃᵃ(i, j, k, grid, f, args...))
@inline δxTᶠᵃᵃ(i, j, k, grid::AGXR{FT}, f, args...) where FT = ifelse(i == 1, zero(FT), δxᶠᵃᵃ(i, j, k, grid, f, args...))

@inline δyTᵃᶠᵃ(i, j, k, grid::AGYB{FT}, f, args...) where FT = ifelse(j == 1, zero(FT), δyᵃᶠᵃ(i, j, k, grid, f, args...))
@inline δyTᵃᶠᵃ(i, j, k, grid::AGYR{FT}, f, args...) where FT = ifelse(j == 1, zero(FT), δyᵃᶠᵃ(i, j, k, grid, f, args...))

# Enforce Impenetrability conditions: `u⁻` vanishes on a left wall and `u⁺` on a right wall
@inline impenetrable_difference(u⁺, u⁻, left_wall, right_wall) = ifelse(right_wall, -u⁻, ifelse(left_wall, u⁺, u⁺ - u⁻))

@inline δxTᶜᵃᵃ(i, j, k, grid::AGXB, f, args...) = impenetrable_difference(f(i + 1, j, k, grid, args...), f(i, j, k, grid, args...), i == 1, i == grid.Nx)
@inline δxTᶜᵃᵃ(i, j, k, grid::AGXL, f, args...) = impenetrable_difference(f(i + 1, j, k, grid, args...), f(i, j, k, grid, args...), false,  i == grid.Nx)
@inline δxTᶜᵃᵃ(i, j, k, grid::AGXR, f, args...) = impenetrable_difference(f(i + 1, j, k, grid, args...), f(i, j, k, grid, args...), i == 1, false)

@inline δyTᵃᶜᵃ(i, j, k, grid::AGYB, f, args...) = impenetrable_difference(f(i, j + 1, k, grid, args...), f(i, j, k, grid, args...), j == 1, j == grid.Ny)
@inline δyTᵃᶜᵃ(i, j, k, grid::AGYL, f, args...) = impenetrable_difference(f(i, j + 1, k, grid, args...), f(i, j, k, grid, args...), false,  j == grid.Ny)
@inline δyTᵃᶜᵃ(i, j, k, grid::AGYR, f, args...) = impenetrable_difference(f(i, j + 1, k, grid, args...), f(i, j, k, grid, args...), j == 1, false)

@inline δxTᶜᵃᵃ(i, j, k, grid::AGXB, u::AbstractArray) = @inbounds impenetrable_difference(u[i + 1, j, k], u[i, j, k], i == 1, i == grid.Nx)
@inline δxTᶜᵃᵃ(i, j, k, grid::AGXL, u::AbstractArray) = @inbounds impenetrable_difference(u[i + 1, j, k], u[i, j, k], false,  i == grid.Nx)
@inline δxTᶜᵃᵃ(i, j, k, grid::AGXR, u::AbstractArray) = @inbounds impenetrable_difference(u[i + 1, j, k], u[i, j, k], i == 1, false)

@inline δyTᵃᶜᵃ(i, j, k, grid::AGYB, v::AbstractArray) = @inbounds impenetrable_difference(v[i, j + 1, k], v[i, j, k], j == 1, j == grid.Ny)
@inline δyTᵃᶜᵃ(i, j, k, grid::AGYL, v::AbstractArray) = @inbounds impenetrable_difference(v[i, j + 1, k], v[i, j, k], false,  j == grid.Ny)
@inline δyTᵃᶜᵃ(i, j, k, grid::AGYR, v::AbstractArray) = @inbounds impenetrable_difference(v[i, j + 1, k], v[i, j, k], j == 1, false)

# Enforce the north fold of a serial tripolar grid: `north_fold_index(i, grid)` mirrors the Center-x column `i` about the fold pivot.
function north_fold_index end

@inline function δyTᵃᶜᵃ(i, j, k, grid::AGYCF, f, args...)
    fold = j == grid.Ny
    v⁺   = f(ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j + 1), k, grid, args...)
    return impenetrable_difference(ifelse(fold, -v⁺, v⁺), f(i, j, k, grid, args...), j == 1, false)
end

@inline function δyTᵃᶜᵃ(i, j, k, grid::AGYCF, v::AbstractArray)
    fold = j == grid.Ny
    v⁺   = @inbounds v[ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j + 1), k]
    return @inbounds impenetrable_difference(ifelse(fold, -v⁺, v⁺), v[i, j, k], j == 1, false)
end

@inline function δyTᵃᶠᵃ(i, j, k, grid::AGYFF{FT}, f, args...) where FT
    fold = j == grid.Ny + 1
    δf   = f(ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j), k, grid, args...) - f(i, j - 1, k, grid, args...)
    return ifelse(j == 1, zero(FT), δf)
end

@inline function δyTᵃᶠᵃ(i, j, k, grid::AGYFF{FT}, c::AbstractArray) where FT
    fold = j == grid.Ny + 1
    δc   = @inbounds c[ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j), k] - c[i, j - 1, k]
    return ifelse(j == 1, zero(FT), δc)
end

@inline function δyTᵃᶜᵃ(i, j, k, grid::AGYFF, f, args...)
    fold = j == grid.Ny + 1
    v⁺   = f(ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j + 1), k, grid, args...)
    return impenetrable_difference(ifelse(fold, -v⁺, v⁺), f(i, j, k, grid, args...), j == 1, false)
end

@inline function δyTᵃᶜᵃ(i, j, k, grid::AGYFF, v::AbstractArray)
    fold = j == grid.Ny + 1
    v⁺   = @inbounds v[ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j + 1), k]
    return @inbounds impenetrable_difference(ifelse(fold, -v⁺, v⁺), v[i, j, k], j == 1, false)
end

# Derivative operators
@inline ∂xTᶠᶜᶠ(i, j, k, grid, f, args...) = δxTᶠᵃᵃ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
@inline ∂yTᶜᶠᶠ(i, j, k, grid, f, args...) = δyTᵃᶠᵃ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)

@inline ∂xTᶠᶜᶠ(i, j, k, grid, w::AbstractArray) = δxTᶠᵃᵃ(i, j, k, grid, w) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
@inline ∂yTᶜᶠᶠ(i, j, k, grid, w::AbstractArray) = δyTᵃᶠᵃ(i, j, k, grid, w) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)
