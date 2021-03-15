
#####
##### Getting minima of grid Δs 
#####

min_Δxyz(grid) = min(min_Δx(grid), min_Δy(grid), min_Δz(grid))
min_Δxy(grid) = min(min_Δx(grid), min_Δy(grid))

min_Δx(grid::RegularRectilinearGrid) = grid.Δx
min_Δy(grid::RegularRectilinearGrid) = grid.Δy
min_Δz(grid::RegularRectilinearGrid) = grid.Δz

min_Δx(grid::VerticallyStretchedRectilinearGrid) = grid.Δx
min_Δy(grid::VerticallyStretchedRectilinearGrid) = grid.Δy
min_Δz(grid::VerticallyStretchedRectilinearGrid) = minimum(grid.Δzᵃᵃᶜ[1:grid.Nz])



