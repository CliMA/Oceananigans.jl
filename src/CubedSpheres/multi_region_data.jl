using Oceananigans.Architectures

import Oceananigans.Grids: new_data

function new_multi_region_data(FT, arch, grid, loc)
    regional_data = Tuple(new_data(FT, arch, regional_grid, loc) for regional_grid in grid.regions)
    return MultiRegionTuple(regional_data)
end

# Resolve ambiguities
new_data(FT, arch::AbstractCPUArchitecture, grid::MultiRegionGrid, loc) = new_multi_region_data(FT, arch, grid, loc)
new_data(FT, arch::AbstractGPUArchitecture, grid::MultiRegionGrid, loc) = new_multi_region_data(FT, arch, grid, loc)

