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

@inline Axᵃᵃᶜ(i, j, k, grid) = Δy(i, j, k, grid) * ΔzF(i, j, k, grid)
@inline Axᵃᵃᶠ(i, j, k, grid) = Δy(i, j, k, grid) * ΔzC(i, j, k, grid)

@inline Ayᵃᵃᶜ(i, j, k, grid) = Δx(i, j, k, grid) * ΔzF(i, j, k, grid)
@inline Ayᵃᵃᶠ(i, j, k, grid) = Δx(i, j, k, grid) * ΔzC(i, j, k, grid)

@inline Azᵃᵃᵃ(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid)

####
#### Volumes
####

@inline Vᵃᵃᶜ(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid) * ΔzF(i, j, k, grid)
@inline Vᵃᵃᶠ(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid) * ΔzC(i, j, k, grid)

