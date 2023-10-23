using Oceananigans.Grids: AbstractUnderlyingGrid

const AGXB = AbstractUnderlyingGrid{<:Any, Bounded}
const AGXP = AbstractUnderlyingGrid{<:Any, Periodic}
const AGXR = AbstractUnderlyingGrid{<:Any, RightConnected}
const AGXL = AbstractUnderlyingGrid{<:Any, LeftConnected}

const AGYB = AbstractUnderlyingGrid{<:Any, <:Any, Bounded}
const AGYP = AbstractUnderlyingGrid{<:Any, <:Any, Periodic}
const AGYR = AbstractUnderlyingGrid{<:Any, <:Any, RightConnected}
const AGYL = AbstractUnderlyingGrid{<:Any, <:Any, LeftConnected}

# Topology-aware Operators with the following convention:
#
#   `δxCᶠᵃᵃ` : Hardcodes `Noflux` or `Periodic` boundary conditions for a (Center, Center, Center) function `c` in x direction 
#   `δyCᵃᶠᵃ` : Hardcodes `Noflux` or `Periodic` boundary conditions for a (Center, Center, Center) function `c` in y direction
#
#   `δxUᶜᵃᵃ` : Hardcodes `NoPenetration` or `Periodic` boundary conditions for a (Face, Center, Center) function `U` in x direction 
#   `δyVᵃᶜᵃ` : Hardcodes `NoPenetration` or `Periodic` boundary conditions for a (Center, Face, Center) function `V` in y direction

# Fallback 

@inline δxCᶠᵃᵃ(i, j, k, grid, c★::Function, args...) = δxᶠᵃᵃ(i, j, k, grid, c★, args...)
@inline δyCᵃᶠᵃ(i, j, k, grid, c★::Function, args...) = δyᵃᶠᵃ(i, j, k, grid, c★, args...)
@inline δxUᶜᵃᵃ(i, j, k, grid, U★::Function, args...) = δxᶜᵃᵃ(i, j, k, grid, U★, args...)
@inline δyVᵃᶜᵃ(i, j, k, grid, V★::Function, args...) = δyᵃᶜᵃ(i, j, k, grid, V★, args...)

# Enforce Periodic conditions for `c★`

@inline δxCᶠᵃᵃ(i, j, k, grid::AGXP, c★::Function, args...) = ifelse(i == 1, c★(1, j, k, grid, args...) - c★(grid.Nx, j, k, grid, args...), δxᶠᵃᵃ(i, j, k, grid, c★, args...))
@inline δyCᵃᶠᵃ(i, j, k, grid::AGYP, c★::Function, args...) = ifelse(j == 1, c★(i, 1, k, grid, args...) - c★(i, grid.Ny, k, grid, args...), δyᵃᶠᵃ(i, j, k, grid, c★, args...))

# Enforce Periodic conditions for `U★` and `V★`

@inline δxUᶜᵃᵃ(i, j, k, grid::AGXP, U★::Function, args...) = ifelse(i == grid.Nx, U★(1, j, k, grid, args...) - U★(grid.Nx, j, k, grid, args...), δxᶜᵃᵃ(i, j, k, grid, U★, args...))
@inline δyVᵃᶜᵃ(i, j, k, grid::AGYP, V★::Function, args...) = ifelse(j == grid.Ny, V★(i, 1, k, grid, args...) - V★(i, grid.Ny, k, grid, args...), δyᵃᶜᵃ(i, j, k, grid, V★, args...))

# Enforce NoFlux conditions for `c★`

@inline δxCᶠᵃᵃ(i, j, k, grid::AGXB, c★::Function, args...) = ifelse(i == 1, 0.0, δxᶠᵃᵃ(i, j, k, grid, c★, args...))
@inline δyCᵃᶠᵃ(i, j, k, grid::AGYB, c★::Function, args...) = ifelse(j == 1, 0.0, δyᵃᶠᵃ(i, j, k, grid, c★, args...))
@inline δxCᶠᵃᵃ(i, j, k, grid::AGXR, c★::Function, args...) = ifelse(i == 1, 0.0, δxᶠᵃᵃ(i, j, k, grid, c★, args...))
@inline δyCᵃᶠᵃ(i, j, k, grid::AGYR, c★::Function, args...) = ifelse(j == 1, 0.0, δyᵃᶠᵃ(i, j, k, grid, c★, args...))

# Enforce Impenetrability conditions for `U★` and `V★`

@inline δxUᶜᵃᵃ(i, j, k, grid::AGXB, U★::Function, args...) = ifelse(i == grid.Nx, - U★(i, j, k, grid, args...),
                                                              ifelse(i == 1, U★(2, j, k, grid, args...), δxᶜᵃᵃ(i, j, k, grid, U★, args...)))
@inline δyVᵃᶜᵃ(i, j, k, grid::AGYB, V★::Function, args...) = ifelse(j == grid.Ny, - V★(i, j, k, grid, args...), 
                                                              ifelse(j == 1, V★(i, 2, k, grid, args...), δyᵃᶜᵃ(i, j, k, grid, V★, args...)))

@inline δxUᶜᵃᵃ(i, j, k, grid::AGXL, U★::Function, args...) = ifelse(i == grid.Nx, - U★(i, j, k, grid, args...), δxᶜᵃᵃ(i, j, k, grid, U★, args...))
@inline δyVᵃᶜᵃ(i, j, k, grid::AGYL, V★::Function, args...) = ifelse(j == grid.Ny, - V★(i, j, k, grid, args...), δyᵃᶜᵃ(i, j, k, grid, V★, args...))

@inline δxUᶜᵃᵃ(i, j, k, grid::AGXR, U★::Function, args...) = ifelse(i == 1, U★(2, j, k, grid, args...), δxᶜᵃᵃ(i, j, k, grid, U★, args...))
@inline δyVᵃᶜᵃ(i, j, k, grid::AGYR, V★::Function, args...) = ifelse(j == 1, V★(i, 2, k, grid, args...), δyᵃᶜᵃ(i, j, k, grid, V★, args...))

# Derivative Operators

@inline ∂xCᶠᶜᶠ(i, j, k, grid, c★::Function, args...) = δxᶠᵃᵃ_c(i, j, k, grid, c★, args...) / Δxᶠᶜᶠ(i, j, k, grid)
@inline ∂yCᶜᶠᶠ(i, j, k, grid, c★::Function, args...) = δyᵃᶠᵃ_c(i, j, k, grid, c★, args...) / Δyᶜᶠᶠ(i, j, k, grid)
                                                   