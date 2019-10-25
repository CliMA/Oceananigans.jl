####
#### Grid spacings
####

@inline Δx(i, j, k, grid) = grid.Δx
@inline Δy(i, j, k, grid) = grid.Δy

@inline ΔzC(i, j, k, grid::RegularCartesianGrid) = grid.Δz
# @inline ΔzC(i, j, k, grid::VerticallyStretchedCartesianGrid) = @inbounds grid.ΔzC[k]

@inline ΔzF(i, j, k, grid::RegularCartesianGrid) = grid.Δz
# @inline ΔzF(i, j, k, grid::VerticallyStretchedCartesianGrid) = @inbounds grid.ΔzF[k]

####
#### Areas
####

@inline Ax_aac(i, j, k, grid) = Δy(i, j, k, grid) * ΔzF(i, j, k, grid)
@inline Ax_aaf(i, j, k, grid) = Δy(i, j, k, grid) * ΔzC(i, j, k, grid)

@inline Ay_aac(i, j, k, grid) = Δx(i, j, k, grid) * ΔzF(i, j, k, grid)
@inline Ay_aaf(i, j, k, grid) = Δx(i, j, k, grid) * ΔzC(i, j, k, grid)

@inline Az(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid)

####
#### Volumes
####

@inline V_aac(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid) * ΔzF(i, j, k, grid)
@inline V_aaf(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid) * ΔzC(i, j, k, grid)

const Vᵘ = V_aac
const Vᵛ = V_aac
const Vʷ = V_aaf
