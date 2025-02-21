module OrthogonalSphericalShellGrids

import Oceananigans
using Oceananigans.Architectures: AbstractArchitecture, CPU, architecture
using Oceananigans.Grids: OrthogonalSphericalShellGrid, R_Earth, halo_size
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

include("displaced_latitude_longitude_grid.jl")

end # module
