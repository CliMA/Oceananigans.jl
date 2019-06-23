# Grid spacing between cells and faces in the x-direction.
@inline Δx(i, j, k, grid::RegularCartesianGrid) = grid.Δx
@inline Δx(i, j, k, grid::VerticallyStretchedCartesianGrid) = grid.Δx
@inline Δx(i, j, k, grid::Grid) = @inbounds grid.Δx[i, j, k]

# Grid spacing between cells and faces in the y-direction.
@inline Δy(i, j, k, grid::RegularCartesianGrid) = grid.Δy
@inline Δy(i, j, k, grid::VerticallyStretchedCartesianGrid) = grid.Δy
@inline Δy(i, j, k, grid::Grid) = @inbounds grid.Δy[i, j, k]

# Grid spacing between cell centers in the z-direction.
@inline ΔzC(i, j, k, grid::RegularCartesianGrid) = grid.Δz
@inline ΔzC(i, j, k, grid::VerticallyStretchedCartesianGrid) = @inbounds grid.ΔzC[k]
@inline ΔzC(i, j, k, grid::Grid) = @inbounds grid.ΔzC[i, j, k]

# Grid spacing between cell faces in the z-direction.
@inline ΔzF(i, j, k, grid::RegularCartesianGrid) = grid.Δz
@inline ΔzF(i, j, k, grid::VerticallyStretchedCartesianGrid) = @inbounds grid.ΔzF[k]
@inline ΔzF(i, j, k, grid::Grid) = @inbounds grid.ΔzF[i, j, k]

# Area of cell faces surrounding cell center and cell face (i, j, k) in the x-direction.
@inline AxC(i, j, k, grid::Grid) = Δy(i, j, k, grid) * ΔzC(i, j, k, grid)
@inline AxF(i, j, k, grid::Grid) = Δy(i, j, k, grid) * ΔzF(i, j, k, grid)

# Area of cell faces surrounding cell center and cell face (i, j, k) in the y-direction.
@inline AyC(i, j, k, grid::Grid) = Δx(i, j, k, grid) * ΔzC(i, j, k, grid)
@inline AyF(i, j, k, grid::Grid) = Δx(i, j, k, grid) * ΔzF(i, j, k, grid)

# Area of cell faces surrounding cell center and cell face (i, j, k) in the z-direction.
@inline Az(i, j, k, grid::Grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid)

# Volume of a cell.
@inline VF(i, j, k, grid::Grid) = ΔxF(i, j, k, grid) * ΔyF(i, j, k, grid) * Δz(i, j, k, grid)
@inline VC(i, j, k, grid::Grid) = ΔxC(i, j, k, grid) * ΔyC(i, j, k, grid) * Δz(i, j, k, grid)

@inline Vᵘ(i, j, k, grid::Grid{T}) where T = T(0.5) * (VF(i, j, k, grid) + VF(i+1, j, k, grid))
@inline Vᵛ(i, j, k, grid::Grid{T}) where T = T(0.5) * (VF(i, j, k, grid) + VF(i, j+1, k, grid))
@inline Vʷ(i, j, k, grid::Grid{T}) where T = T(0.5) * (VF(i, j, k, grid) + VF(i, j, k+1, grid))
