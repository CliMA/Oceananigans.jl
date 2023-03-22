using Oceananigans.AbstractOperations: GridMetricOperation, Δz
using Oceananigans.Distributed: DistributedGrid
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitState, SplitExplicitFreeSurface

import Oceananigans.Models.HydrostaticFreeSurfaceModels: FreeSurface, SplitExplicitAuxiliaryFields

function SplitExplicitAuxiliaryFields(grid::DistributedGrid)
    
    Gᵁ = Field((Face,   Center, Nothing), grid)
    Gⱽ = Field((Center, Face,   Nothing), grid)
    
    Hᶠᶜ = Field((Face,   Center, Nothing), grid)
    Hᶜᶠ = Field((Center, Face,   Nothing), grid)
    Hᶜᶜ = Field((Center, Center, Nothing), grid)
    
    calculate_column_height!(Hᶠᶜ, (Face, Center, Center))
    calculate_column_height!(Hᶜᶠ, (Center, Face, Center))
    calculate_column_height!(Hᶜᶜ, (Center, Center, Center))
       
    fill_halo_regions!((Hᶠᶜ, Hᶜᶠ, Hᶜᶜ))

    # In a non-parallel grid we calculate only the interior
    kernel_size    = augmented_kernel_size(grid)
    kernel_offsets = augmented_kernel_offsets(grid)
    
    return SplitExplicitAuxiliaryFields(Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ, Hᶜᶜ, kernel_size, kernel_offsets)
end

"""Integrate z at locations `location`."""
@inline function calculate_column_height!(height, location)
    dz = GridMetricOperation(location, Δz, height.grid)
    return sum!(height, dz)
end

@inline function augmented_kernel_size(grid::DistributedGrid)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    Tx, Ty, _ = topology(grid)
    Rx, Ry, _ = architecture(grid).ranks

    Ax = Rx == 1 ? Nx : (Tx == RightConnected || Tx == LeftConnected ? Nx + Hx - 1 : Nx + 2Hx - 2)
    Ay = Ry == 1 ? Ny : (Ty == RightConnected || Ty == LeftConnected ? Ny + Hy - 1 : Nx + 2Hy - 2)

    return (Ax, Ay)
end
   
@inline function augmented_kernel_offsets(grid::DistributedGrid)
    Hx, Hy, _ = halo_size(grid)
    Tx, Ty, _ = topology(grid)
    Rx, Ry, _ = architecture(grid).ranks

    Ax = Rx == 1 || Tx == RightConnected ? 0 : Hx - 1
    Ay = Ry == 1 || Ty == RightConnected ? 0 : Hy - 1

    return (Ax, Ay)
end

function FreeSurface(free_surface::SplitExplicitFreeSurface, velocities, grid::DistributedGrid)
    settings  = free_surface.settings 
    current_halo = halo_size(grid)

    # Build an expanded "split-explicit grid" with (potentially) huge halos
    # so we can avoid communication during split-explicit substepping
    se_halo = split_explicit_halo(current_halo, settings.substeps, grid)         
    se_grid = with_halo(se_halo, grid)
    
    Nz = size(se_grid, 3)
    η = ZFaceField(se_grid, indices=(:, :, Nz+1))

    return SplitExplicitFreeSurface(η,
                                    SplitExplicitState(new_grid),
                                    SplitExplicitAuxiliaryFields(new_grid),
                                    free_surface.gravitational_acceleration,
                                    free_surface.settings)
end

@inline function split_explicit_halo(current_halo, Nsubsteps, grid::DistributedGrid)
    arch = architecture(grid)
    Rx, Ry, _ = arch.ranks

    # Inflate halos given current number of substeps
    Hx = Rx == 1 ? current_halo[1] : Nsubsteps + 1
    Hy = Ry == 1 ? current_halo[2] : Nsubsteps + 1

    return (Hx, Hy, current_halo[3])
end
