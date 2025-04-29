using Oceananigans.Grids: AbstractUnderlyingGrid

const AGXB = AbstractUnderlyingGrid{FT, Bounded} where FT
const AGXP = AbstractUnderlyingGrid{FT, Periodic} where FT
const AGXR = AbstractUnderlyingGrid{FT, RightConnected} where FT
const AGXL = AbstractUnderlyingGrid{FT, LeftConnected} where FT

const AGYB = AbstractUnderlyingGrid{FT, <:Any, Bounded} where FT
const AGYP = AbstractUnderlyingGrid{FT, <:Any, Periodic} where FT
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
@inline δxTᶠᵃᵃ(i, j, k, grid, f, args...) = δxᶠᵃᵃ(i, j, k, grid, f, args...)
@inline δyTᵃᶠᵃ(i, j, k, grid, f, args...) = δyᵃᶠᵃ(i, j, k, grid, f, args...)
@inline δxTᶜᵃᵃ(i, j, k, grid, f, args...) = δxᶜᵃᵃ(i, j, k, grid, f, args...)
@inline δyTᵃᶜᵃ(i, j, k, grid, f, args...) = δyᵃᶜᵃ(i, j, k, grid, f, args...)

# Enforce Periodic conditions
@inline δxTᶠᵃᵃ(i, j, k, grid::AGXP, f, args...) = ifelse(i == 1, f(1, j, k, grid, args...) - f(grid.Nx, j, k, grid, args...), δxᶠᵃᵃ(i, j, k, grid, f, args...))
@inline δyTᵃᶠᵃ(i, j, k, grid::AGYP, f, args...) = ifelse(j == 1, f(i, 1, k, grid, args...) - f(i, grid.Ny, k, grid, args...), δyᵃᶠᵃ(i, j, k, grid, f, args...))

@inline δxTᶠᵃᵃ(i, j, k, grid::AGXP, c::AbstractArray) = @inbounds ifelse(i == 1, c[1, j, k] - c[grid.Nx, j, k], δxᶠᵃᵃ(i, j, k, grid, c))
@inline δyTᵃᶠᵃ(i, j, k, grid::AGYP, c::AbstractArray) = @inbounds ifelse(j == 1, c[i, 1, k] - c[i, grid.Ny, k], δyᵃᶠᵃ(i, j, k, grid, c))

@inline δxTᶜᵃᵃ(i, j, k, grid::AGXP, f, args...) = ifelse(i == grid.Nx, f(1, j, k, grid, args...) - f(grid.Nx, j, k, grid, args...), δxᶜᵃᵃ(i, j, k, grid, f, args...))
@inline δyTᵃᶜᵃ(i, j, k, grid::AGYP, f, args...) = ifelse(j == grid.Ny, f(i, 1, k, grid, args...) - f(i, grid.Ny, k, grid, args...), δyᵃᶜᵃ(i, j, k, grid, f, args...))

@inline δxTᶜᵃᵃ(i, j, k, grid::AGXP, u::AbstractArray) = @inbounds ifelse(i == grid.Nx, u[1, j, k] - u[grid.Nx, j, k], δxᶜᵃᵃ(i, j, k, grid, u))
@inline δyTᵃᶜᵃ(i, j, k, grid::AGYP, v::AbstractArray) = @inbounds ifelse(j == grid.Ny, v[i, 1, k] - v[i, grid.Ny, k], δyᵃᶜᵃ(i, j, k, grid, v))

# Enforce NoFlux conditions
@inline δxTᶠᵃᵃ(i, j, k, grid::AGXB{FT}, f, args...) where FT = ifelse(i == 1, zero(FT), δxᶠᵃᵃ(i, j, k, grid, f, args...))
@inline δyTᵃᶠᵃ(i, j, k, grid::AGYB{FT}, f, args...) where FT = ifelse(j == 1, zero(FT), δyᵃᶠᵃ(i, j, k, grid, f, args...))

@inline δxTᶠᵃᵃ(i, j, k, grid::AGXR{FT}, f, args...) where FT = ifelse(i == 1, zero(FT), δxᶠᵃᵃ(i, j, k, grid, f, args...))
@inline δyTᵃᶠᵃ(i, j, k, grid::AGYR{FT}, f, args...) where FT = ifelse(j == 1, zero(FT), δyᵃᶠᵃ(i, j, k, grid, f, args...))

# Enforce Impenetrability conditions
@inline δxTᶜᵃᵃ(i, j, k, grid::AGXB, f, args...) =
    ifelse(i == grid.Nx, - f(i, j, k, grid, args...),
    ifelse(i == 1, f(2, j, k, grid, args...), δxᶜᵃᵃ(i, j, k, grid, f, args...)))

@inline δxTᶜᵃᵃ(i, j, k, grid::AGXB, u::AbstractArray) =
    @inbounds ifelse(i == grid.Nx, - u[i, j, k],
              ifelse(i == 1, u[2, j, k], δxᶜᵃᵃ(i, j, k, grid, u)))

@inline δyTᵃᶜᵃ(i, j, k, grid::AGYB, f, args...) =
    ifelse(j == grid.Ny, - f(i, j, k, grid, args...),
    ifelse(j == 1, f(i, 2, k, grid, args...), δyᵃᶜᵃ(i, j, k, grid, f, args...)))

@inline δyTᵃᶜᵃ(i, j, k, grid::AGYB, v::AbstractArray) =
    @inbounds ifelse(j == grid.Ny, - v[i, j, k],
              ifelse(j == 1, v[i, 2, k], δyᵃᶜᵃ(i, j, k, grid, v)))

@inline δxTᶜᵃᵃ(i, j, k, grid::AGXL, f, args...) = ifelse(i == grid.Nx, - f(i, j, k, grid, args...), δxᶜᵃᵃ(i, j, k, grid, f, args...))
@inline δyTᵃᶜᵃ(i, j, k, grid::AGYL, f, args...) = ifelse(j == grid.Ny, - f(i, j, k, grid, args...), δyᵃᶜᵃ(i, j, k, grid, f, args...))

@inline δxTᶜᵃᵃ(i, j, k, grid::AGXL, u::AbstractArray) = @inbounds ifelse(i == grid.Nx, - u[i, j, k], δxᶜᵃᵃ(i, j, k, grid, u))
@inline δyTᵃᶜᵃ(i, j, k, grid::AGYL, v::AbstractArray) = @inbounds ifelse(j == grid.Ny, - v[i, j, k], δyᵃᶜᵃ(i, j, k, grid, v))

@inline δxTᶜᵃᵃ(i, j, k, grid::AGXR, f, args...) = ifelse(i == 1, f(2, j, k, grid, args...), δxᶜᵃᵃ(i, j, k, grid, f, args...))
@inline δyTᵃᶜᵃ(i, j, k, grid::AGYR, f, args...) = ifelse(j == 1, f(i, 2, k, grid, args...), δyᵃᶜᵃ(i, j, k, grid, f, args...))

@inline δxTᶜᵃᵃ(i, j, k, grid::AGXR, u::AbstractArray) = @inbounds ifelse(i == 1, u[2, j, k], δxᶜᵃᵃ(i, j, k, grid, u))
@inline δyTᵃᶜᵃ(i, j, k, grid::AGYR, v::AbstractArray) = @inbounds ifelse(j == 1, v[i, 2, k], δyᵃᶜᵃ(i, j, k, grid, v))

# Derivative operators
@inline ∂xTᶠᶜᶠ(i, j, k, grid, f, args...) = δxTᶠᵃᵃ(i, j, k, grid, f, args...) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
@inline ∂yTᶜᶠᶠ(i, j, k, grid, f, args...) = δyTᵃᶠᵃ(i, j, k, grid, f, args...) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)

@inline ∂xTᶠᶜᶠ(i, j, k, grid, w::AbstractArray) = δxTᶠᵃᵃ(i, j, k, grid, w) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)
@inline ∂yTᶜᶠᶠ(i, j, k, grid, w::AbstractArray) = δyTᵃᶠᵃ(i, j, k, grid, w) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)
