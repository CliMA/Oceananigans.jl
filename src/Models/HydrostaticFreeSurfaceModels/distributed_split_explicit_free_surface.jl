using Oceananigans.AbstractOperations: GridMetricOperation, Δz
using Oceananigans.Distributed: DistributedGrid
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitState, SplitExplicitFreeSurface

import Oceananigans.Models.HydrostaticFreeSurfaceModels: FreeSurface, SplitExplicitAuxiliary

function SplitExplicitAuxiliary(grid::DistributedGrid)
    
    Gᵁ = Field((Face,   Center, Nothing), grid)
    Gⱽ = Field((Center, Face,   Nothing), grid)
    
    Hᶠᶜ = Field((Face,   Center, Nothing), grid)
    Hᶜᶠ = Field((Center, Face,   Nothing), grid)
    Hᶜᶜ = Field((Center, Center, Nothing), grid)
    
    vertical_height!(Hᶠᶜ, (Face, Center, Center))
    vertical_height!(Hᶜᶠ, (Center, Face, Center))

    vertical_height!(Hᶜᶜ, (Center, Center, Center))
       
    fill_halo_regions!((Hᶠᶜ, Hᶜᶠ, Hᶜᶜ))

    # In a non-parallel grid we calculate only the interior
    kernel_size    = augmented_kernel_size(grid)
    kernel_offsets = augmented_kernel_offsets(grid)
    
    return SplitExplicitAuxiliary(Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ, Hᶜᶜ, kernel_size, kernel_offsets)
end

@inline function vertical_height!(height, location)
    dz = GridMetricOperation(location, Δz, height.grid)
    sum!(height, dz)
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

        old_halos = halo_size(grid)

        new_halos = split_explicit_halos(old_halos, settings.substeps+1, grid)         
        new_grid  = with_halo(new_halos, grid)
    
        η = ZFaceField(new_grid, indices = (:, :, size(new_grid, 3)+1))

        return SplitExplicitFreeSurface(η,
                                        SplitExplicitState(new_grid),
                                        SplitExplicitAuxiliary(new_grid),
                                        free_surface.gravitational_acceleration,
                                        free_surface.settings)
end

@inline function split_explicit_halos(old_halos, step_halo, grid::DistributedGrid)

    Rx, Ry, _ = architecture(grid).ranks

    Ax = Rx == 1 ? old_halos[1] : step_halo
    Ay = Ry == 1 ? old_halos[2] : step_halo

    return (Ax, Ay, old_halos[3])
end
