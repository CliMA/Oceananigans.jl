# Width of the control volumes in the x-direction.
@inline Δx(i, j, k, grid::AbstractGrid) = grid.Δx

# Width of the control volumes in the y-direction.
@inline Δy(i, j, k, grid::AbstractGrid) = grid.Δy

# Height of the control volumes containing the cell centers in the z-direction.
@inline ΔzC(i, j, k, grid::RegularCartesianGrid) = grid.Δz
@inline ΔzC(i, j, k, grid::VerticallyStretchedCartesianGrid) = @inbounds grid.ΔzC[i]

# Height of the control volumes containing the cell w-faces in the z-direction.
@inline ΔzF(i, j, k, grid::RegularCartesianGrid) = grid.Δz
@inline ΔzF(i, j, k, grid::VerticallyStretchedCartesianGrid) = @inbounds grid.ΔzF[k]

# Area of control volume faces surrounding cell center and cell face (i, j, k) in the x-direction
@inline AxC(i, j, k, grid) = Δy(i, j, k, grid) * ΔzC(i, j, k, grid)
@inline AxF(i, j, k, grid) = Δy(i, j, k, grid) * ΔzF(i, j, k, grid)

# Area of control volume faces surrounding cell center and cell face (i, j, k) in the y-direction
@inline AyC(i, j, k, grid) = Δx(i, j, k, grid) * ΔzC(i, j, k, grid)
@inline AyF(i, j, k, grid) = Δx(i, j, k, grid) * ΔzF(i, j, k, grid)

# Area of control volume faces surrounding cell center and cell face (i, j, k) in the z-direction
@inline Az(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid)

# Volume of a control volume surrounding a cell center.
@inline Vᶜ(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid) * ΔzC(i, j, k, grid)

# Volume of a control volume surrounding the cell u-, v-, and w-faces.
@inline Vᵘ(i, j, k, grid) = Vᶜ(i, j, k, grid)
@inline Vᵛ(i, j, k, grid) = Vᶜ(i, k, k, grid)
@inline Vʷ(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid) * ΔzF(i, j, k, grid)

