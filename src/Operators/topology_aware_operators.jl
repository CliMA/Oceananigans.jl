using Oceananigans.Grids: AbstractUnderlyingGrid

const AGXB = AbstractUnderlyingGrid{FT, Bounded} where FT
const AGXP = AbstractUnderlyingGrid{FT, Periodic} where FT
const AGXF = AbstractUnderlyingGrid{FT, RightFolded} where FT
const AGXR = AbstractUnderlyingGrid{FT, RightConnected} where FT
const AGXL = AbstractUnderlyingGrid{FT, LeftConnected} where FT

const AGYB = AbstractUnderlyingGrid{FT, <:Any, Bounded} where FT
const AGYP = AbstractUnderlyingGrid{FT, <:Any, Periodic} where FT
const AGYF = AbstractUnderlyingGrid{FT, <:Any, RightFolded} where FT
const AGYR = AbstractUnderlyingGrid{FT, <:Any, RightConnected} where FT
const AGYL = AbstractUnderlyingGrid{FT, <:Any, LeftConnected} where FT

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

@inline δxTᶠᶜᶠ(i, j, k, grid, f::Function, args...) = δxᶠᵃᵃ(i, j, k, grid, f, args...)
@inline δyTᶜᶠᶠ(i, j, k, grid, f::Function, args...) = δyᵃᶠᵃ(i, j, k, grid, f, args...)
@inline δxTᶜᶜᶠ(i, j, k, grid, f::Function, args...) = δxᶜᵃᵃ(i, j, k, grid, f, args...)
@inline δyTᶜᶜᶠ(i, j, k, grid, f::Function, args...) = δyᵃᶜᵃ(i, j, k, grid, f, args...)

# Enforce Periodic conditions

@inline δxTᶠᶜᵃ(i, j, k, grid::AGXP, f::Function, args...) = ifelse(i == 1, f(1, j, k, grid, args...) - f(grid.Nx, j, k, grid, args...), δxᶠᵃᵃ(i, j, k, grid, f, args...))
@inline δyTᶠᵃ(i, j, k, grid::AGYP, f::Function, args...) = ifelse(j == 1, f(i, 1, k, grid, args...) - f(i, grid.Ny, k, grid, args...), δyᵃᶠᵃ(i, j, k, grid, f, args...))

@inline δxTᶜᵃᵃ(i, j, k, grid::AGXP, f::Function, args...) = ifelse(i == grid.Nx, f(1, j, k, grid, args...) - f(grid.Nx, j, k, grid, args...), δxᶜᵃᵃ(i, j, k, grid, f, args...))
@inline δyTᵃᶜᵃ(i, j, k, grid::AGYP, f::Function, args...) = ifelse(j == grid.Ny, f(i, 1, k, grid, args...) - f(i, grid.Ny, k, grid, args...), δyᵃᶜᵃ(i, j, k, grid, f, args...))

# Enforce NoFlux conditions

@inline δxTᶠᵃᵃ(i, j, k, grid::AGXB{FT}, f::Function, args...) where FT = ifelse(i == 1, zero(FT), δxᶠᵃᵃ(i, j, k, grid, f, args...))
@inline δyTᵃᶠᵃ(i, j, k, grid::AGYB{FT}, f::Function, args...) where FT = ifelse(j == 1, zero(FT), δyᵃᶠᵃ(i, j, k, grid, f, args...))

@inline δxTᶠᵃᵃ(i, j, k, grid::AGXF{FT}, f::Function, args...) where FT = ifelse(i == 1, zero(FT), δxᶠᵃᵃ(i, j, k, grid, f, args...))
@inline δyTᵃᶠᵃ(i, j, k, grid::AGYF{FT}, f::Function, args...) where FT = ifelse(j == 1, zero(FT), δyᵃᶠᵃ(i, j, k, grid, f, args...))

@inline δxTᶠᵃᵃ(i, j, k, grid::AGXR{FT}, f::Function, args...) where FT = ifelse(i == 1, zero(FT), δxᶠᵃᵃ(i, j, k, grid, f, args...))
@inline δyTᵃᶠᵃ(i, j, k, grid::AGYR{FT}, f::Function, args...) where FT = ifelse(j == 1, zero(FT), δyᵃᶠᵃ(i, j, k, grid, f, args...))

# Enforce Impenetrability conditions

@inline δxTᶜᵃᵃ(i, j, k, grid::AGXB, f::Function, args...) =
    ifelse(i == grid.Nx, - f(i, j, k, grid, args...),
                         ifelse(i == 1, f(2, j, k, grid, args...),
                                        δxᶜᵃᵃ(i, j, k, grid, f, args...)))

@inline δyTᵃᶜᵃ(i, j, k, grid::AGYB, f::Function, args...) =
    ifelse(j == grid.Ny, - f(i, j, k, grid, args...),
                         ifelse(j == 1, f(i, 2, k, grid, args...),
                                        δyᵃᶜᵃ(i, j, k, grid, f, args...)))

# Changes across the fold, this operator works for `Center` fields in the x-direction.
@inline δyTᵃᶜᵃ(i, j, k, grid::AGYF, f::Function, args...) = 
    ifelse(j == grid.Ny, folded_δyᵃᶜᵃ(i, j, k, grid, f, args...), δyᵃᶜᵃ(i, j, k, grid, f, args...))

@inline function folded_δyᵃᶜᵃ(i, j, k, grid, f, args...)
    # Retrieve the folded index
    i′ = grid.Nx - i + 1 

    # We switch the sign of the function value at the folded index 
    f₂ = - f(i′, j, k, grid, args...)
    f₁ =   f(i,  j, k, grid, args...)

    return f₂ - f₁
end

# Derivative operators

@inline ∂xTᶠᶜᶠ(i, j, k, grid, f::Function, args...) = δxTᶠᵃᵃ(i, j, k, grid, f, args...) / Δxᶠᶜᶠ(i, j, k, grid)
@inline ∂yTᶜᶠᶠ(i, j, k, grid, f::Function, args...) = δyTᵃᶠᵃ(i, j, k, grid, f, args...) / Δyᶜᶠᶠ(i, j, k, grid)
