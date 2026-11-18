using Oceananigans.Grids: AbstractGrid, Closed, CenterFolded, FaceFolded, Periodic, TwoSided

# These aliases cover every `AbstractGrid` with the given topology, including `ImmersedBoundaryGrid`,
# so an immersed grid dispatches to the topology-aware methods directly and `f` is evaluated on the
# immersed grid rather than on its underlying grid.
const AGXP  = AbstractGrid{FT, Periodic} where FT
const AGXS  = AbstractGrid{FT, TwoSided{L, R}} where {FT, L, R}

const AGYP  = AbstractGrid{FT, <:Any, Periodic} where FT
const AGYS  = AbstractGrid{FT, <:Any, TwoSided{L, R}} where {FT, L, R}
const AGYCF = AbstractGrid{FT, <:Any, TwoSided{L, CenterFolded}} where {FT, L}
const AGYFF = AbstractGrid{FT, <:Any, TwoSided{L, FaceFolded}} where {FT, L}

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

# Enforce NoFlux conditions on a left wall (i = 1) and a right wall (i = Nx + 1)
@inline δxTᶠᵃᵃ(i, j, k, grid::AGXS{FT, L, R}, f, args...) where {FT, L, R} =
    ifelse(((L === Closed) & (i == 1)) | ((R === Closed) & (i == grid.Nx + 1)), zero(FT), δxᶠᵃᵃ(i, j, k, grid, f, args...))

@inline δyTᵃᶠᵃ(i, j, k, grid::AGYS{FT, L, R}, f, args...) where {FT, L, R} =
    ifelse(((L === Closed) & (j == 1)) | ((R === Closed) & (j == grid.Ny + 1)), zero(FT), δyᵃᶠᵃ(i, j, k, grid, f, args...))

# Enforce Impenetrability conditions: `u⁻` vanishes on a left wall and `u⁺` on a right wall
@inline impenetrable_difference(u⁺, u⁻, left_wall, right_wall) = ifelse(right_wall, -u⁻, ifelse(left_wall, u⁺, u⁺ - u⁻))

@inline δxTᶜᵃᵃ(i, j, k, grid::AGXS{<:Any, L, R}, f, args...) where {L, R} = impenetrable_difference(f(i + 1, j, k, grid, args...), f(i, j, k, grid, args...),
                                                                                                    (L === Closed) & (i == 1), (R === Closed) & (i == grid.Nx))

@inline δyTᵃᶜᵃ(i, j, k, grid::AGYS{<:Any, L, R}, f, args...) where {L, R} = impenetrable_difference(f(i, j + 1, k, grid, args...), f(i, j, k, grid, args...),
                                                                                                    (L === Closed) & (j == 1), (R === Closed) & (j == grid.Ny))

@inline δxTᶜᵃᵃ(i, j, k, grid::AGXS{<:Any, L, R}, u::AbstractArray) where {L, R} = @inbounds impenetrable_difference(u[i + 1, j, k], u[i, j, k],
                                                                                                                   (L === Closed) & (i == 1), (R === Closed) & (i == grid.Nx))

@inline δyTᵃᶜᵃ(i, j, k, grid::AGYS{<:Any, L, R}, v::AbstractArray) where {L, R} = @inbounds impenetrable_difference(v[i, j + 1, k], v[i, j, k],
                                                                                                                   (L === Closed) & (j == 1), (R === Closed) & (j == grid.Ny))

# Enforce a north fold local to the rank: `north_fold_index(i, grid)` mirrors the Center-x column `i` about the fold pivot.
function north_fold_index end

@inline function δyTᵃᶜᵃ(i, j, k, grid::AGYCF{<:Any, L}, f, args...) where L
    fold = j == grid.Ny
    v⁺   = f(ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j + 1), k, grid, args...)
    return impenetrable_difference(ifelse(fold, -v⁺, v⁺), f(i, j, k, grid, args...), (L === Closed) & (j == 1), false)
end

@inline function δyTᵃᶜᵃ(i, j, k, grid::AGYCF{<:Any, L}, v::AbstractArray) where L
    fold = j == grid.Ny
    v⁺   = @inbounds v[ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j + 1), k]
    return @inbounds impenetrable_difference(ifelse(fold, -v⁺, v⁺), v[i, j, k], (L === Closed) & (j == 1), false)
end

@inline function δyTᵃᶠᵃ(i, j, k, grid::AGYFF{FT, L}, f, args...) where {FT, L}
    fold = j == grid.Ny + 1
    δf   = f(ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j), k, grid, args...) - f(i, j - 1, k, grid, args...)
    return ifelse((L === Closed) & (j == 1), zero(FT), δf)
end

@inline function δyTᵃᶠᵃ(i, j, k, grid::AGYFF{FT, L}, c::AbstractArray) where {FT, L}
    fold = j == grid.Ny + 1
    δc   = @inbounds c[ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j), k] - c[i, j - 1, k]
    return ifelse((L === Closed) & (j == 1), zero(FT), δc)
end

@inline function δyTᵃᶜᵃ(i, j, k, grid::AGYFF{<:Any, L}, f, args...) where L
    fold = j == grid.Ny + 1
    v⁺   = f(ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j + 1), k, grid, args...)
    return impenetrable_difference(ifelse(fold, -v⁺, v⁺), f(i, j, k, grid, args...), (L === Closed) & (j == 1), false)
end

@inline function δyTᵃᶜᵃ(i, j, k, grid::AGYFF{<:Any, L}, v::AbstractArray) where L
    fold = j == grid.Ny + 1
    v⁺   = @inbounds v[ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j + 1), k]
    return @inbounds impenetrable_difference(ifelse(fold, -v⁺, v⁺), v[i, j, k], (L === Closed) & (j == 1), false)
end

# Topology-aware interpolations with the same convention:
#
#   `ℑxTᶠᵃᵃ` : Interpolates a (Center, Center, Center) function `f` to faces in the x-direction, wrapping across
#              `Periodic` boundaries and returning the adjacent cell value at a `Closed` left wall (zero-gradient,
#              consistent with `δxTᶠᵃᵃ` vanishing there).
#   `ℑxTᶜᵃᵃ` : Interpolates a (Face, Center, Center) function `U` to centers in the x-direction, wrapping across
#              `Periodic` boundaries. The right-wall face is interior data on `TwoSided` grids, so needs no treatment.
#
#   `ℑyTᵃᶠᵃ`, `ℑyTᵃᶜᵃ` : the same in the y-direction, also mirroring across a tripolar north fold.

# Fallback
@inline ℑxTᶠᵃᵃ(i, j, k, grid, f, args...) = ℑxᶠᵃᵃ(i, j, k, grid, f, args...)
@inline ℑyTᵃᶠᵃ(i, j, k, grid, f, args...) = ℑyᵃᶠᵃ(i, j, k, grid, f, args...)
@inline ℑxTᶜᵃᵃ(i, j, k, grid, f, args...) = ℑxᶜᵃᵃ(i, j, k, grid, f, args...)
@inline ℑyTᵃᶜᵃ(i, j, k, grid, f, args...) = ℑyᵃᶜᵃ(i, j, k, grid, f, args...)

# Enforce Periodic conditions
@inline ℑxTᶠᵃᵃ(i, j, k, grid::AGXP{FT}, f, args...) where FT = FT(0.5) * (f(ifelse(i == 1, grid.Nx, i - 1), j, k, grid, args...) + f(i, j, k, grid, args...))
@inline ℑyTᵃᶠᵃ(i, j, k, grid::AGYP{FT}, f, args...) where FT = FT(0.5) * (f(i, ifelse(j == 1, grid.Ny, j - 1), k, grid, args...) + f(i, j, k, grid, args...))

@inline ℑxTᶠᵃᵃ(i, j, k, grid::AGXP{FT}, c::AbstractArray) where FT = @inbounds FT(0.5) * (c[ifelse(i == 1, grid.Nx, i - 1), j, k] + c[i, j, k])
@inline ℑyTᵃᶠᵃ(i, j, k, grid::AGYP{FT}, c::AbstractArray) where FT = @inbounds FT(0.5) * (c[i, ifelse(j == 1, grid.Ny, j - 1), k] + c[i, j, k])

@inline ℑxTᶜᵃᵃ(i, j, k, grid::AGXP{FT}, f, args...) where FT = FT(0.5) * (f(i, j, k, grid, args...) + f(ifelse(i == grid.Nx, 1, i + 1), j, k, grid, args...))
@inline ℑyTᵃᶜᵃ(i, j, k, grid::AGYP{FT}, f, args...) where FT = FT(0.5) * (f(i, j, k, grid, args...) + f(i, ifelse(j == grid.Ny, 1, j + 1), k, grid, args...))

@inline ℑxTᶜᵃᵃ(i, j, k, grid::AGXP{FT}, u::AbstractArray) where FT = @inbounds FT(0.5) * (u[i, j, k] + u[ifelse(i == grid.Nx, 1, i + 1), j, k])
@inline ℑyTᵃᶜᵃ(i, j, k, grid::AGYP{FT}, v::AbstractArray) where FT = @inbounds FT(0.5) * (v[i, j, k] + v[i, ifelse(j == grid.Ny, 1, j + 1), k])

# Enforce zero-gradient conditions on a left wall (i = 1) and a right wall (i = Nx + 1): the image of
# the cell-centered `f` across a `Closed` wall is the wall cell itself
@inline ℑxTᶠᵃᵃ(i, j, k, grid::AGXS{FT, L, R}, f, args...) where {FT, L, R} =
    FT(0.5) * (f(ifelse((L === Closed) & (i == 1), i, i - 1), j, k, grid, args...) +
               f(ifelse((R === Closed) & (i == grid.Nx + 1), i - 1, i), j, k, grid, args...))

@inline ℑyTᵃᶠᵃ(i, j, k, grid::AGYS{FT, L, R}, f, args...) where {FT, L, R} =
    FT(0.5) * (f(i, ifelse((L === Closed) & (j == 1), j, j - 1), k, grid, args...) +
               f(i, ifelse((R === Closed) & (j == grid.Ny + 1), j - 1, j), k, grid, args...))

@inline ℑxTᶠᵃᵃ(i, j, k, grid::AGXS{FT, L, R}, c::AbstractArray) where {FT, L, R} =
    @inbounds FT(0.5) * (c[ifelse((L === Closed) & (i == 1), i, i - 1), j, k] + c[ifelse((R === Closed) & (i == grid.Nx + 1), i - 1, i), j, k])

@inline ℑyTᵃᶠᵃ(i, j, k, grid::AGYS{FT, L, R}, c::AbstractArray) where {FT, L, R} =
    @inbounds FT(0.5) * (c[i, ifelse((L === Closed) & (j == 1), j, j - 1), k] + c[i, ifelse((R === Closed) & (j == grid.Ny + 1), j - 1, j), k])

# Enforce a north fold local to the rank, mirroring `δyTᵃᶜᵃ` and `δyTᵃᶠᵃ` above
@inline function ℑyTᵃᶜᵃ(i, j, k, grid::AGYCF{FT}, f, args...) where FT
    fold = j == grid.Ny
    v⁺   = f(ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j + 1), k, grid, args...)
    return FT(0.5) * (f(i, j, k, grid, args...) + ifelse(fold, -v⁺, v⁺))
end

@inline function ℑyTᵃᶜᵃ(i, j, k, grid::AGYCF{FT}, v::AbstractArray) where FT
    fold = j == grid.Ny
    v⁺   = @inbounds v[ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j + 1), k]
    return @inbounds FT(0.5) * (v[i, j, k] + ifelse(fold, -v⁺, v⁺))
end

@inline function ℑyTᵃᶠᵃ(i, j, k, grid::AGYFF{FT, L}, f, args...) where {FT, L}
    fold = j == grid.Ny + 1
    c⁺   = f(ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j), k, grid, args...)
    c⁻   = f(i, ifelse((L === Closed) & (j == 1), j, j - 1), k, grid, args...)
    return FT(0.5) * (c⁺ + c⁻)
end

@inline function ℑyTᵃᶠᵃ(i, j, k, grid::AGYFF{FT, L}, c::AbstractArray) where {FT, L}
    fold = j == grid.Ny + 1
    c⁺   = @inbounds c[ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j), k]
    c⁻   = @inbounds c[i, ifelse((L === Closed) & (j == 1), j, j - 1), k]
    return FT(0.5) * (c⁺ + c⁻)
end

@inline function ℑyTᵃᶜᵃ(i, j, k, grid::AGYFF{FT}, f, args...) where FT
    fold = j == grid.Ny + 1
    v⁺   = f(ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j + 1), k, grid, args...)
    return FT(0.5) * (f(i, j, k, grid, args...) + ifelse(fold, -v⁺, v⁺))
end

@inline function ℑyTᵃᶜᵃ(i, j, k, grid::AGYFF{FT}, v::AbstractArray) where FT
    fold = j == grid.Ny + 1
    v⁺   = @inbounds v[ifelse(fold, north_fold_index(i, grid), i), ifelse(fold, grid.Ny, j + 1), k]
    return @inbounds FT(0.5) * (v[i, j, k] + ifelse(fold, -v⁺, v⁺))
end

# Derivative operators
@inline ∂xTᶠᶜᶠ(i, j, k, grid, f, args...) = δxTᶠᵃᵃ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
@inline ∂yTᶜᶠᶠ(i, j, k, grid, f, args...) = δyTᵃᶠᵃ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)

@inline ∂xTᶠᶜᶠ(i, j, k, grid, w::AbstractArray) = δxTᶠᵃᵃ(i, j, k, grid, w) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
@inline ∂yTᶜᶠᶠ(i, j, k, grid, w::AbstractArray) = δyTᵃᶠᵃ(i, j, k, grid, w) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)
