using Oceananigans.AbstractOperations: GridMetricOperation, Δz
using Oceananigans.DistributedComputations: DistributedGrid, DistributedField
using Oceananigans.DistributedComputations: SynchronizedDistributed, synchronize_communication!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitState, SplitExplicitFreeSurface

import Oceananigans.Models.HydrostaticFreeSurfaceModels: FreeSurface, SplitExplicitAuxiliaryFields

function SplitExplicitAuxiliaryFields(grid::DistributedGrid)
    
    Nz = size(grid, 3)

    Gᵁ = XFaceField(grid, indices = (:, :, Nz))
    Gⱽ = YFaceField(grid, indices = (:, :, Nz))
    
    # In a non-parallel grid we calculate only the interior
    kernel_size    = augmented_kernel_size(grid)
    kernel_offsets = augmented_kernel_offsets(grid)

    kernel_parameters = KernelParameters(kernel_size, kernel_offsets)
    
    return SplitExplicitAuxiliaryFields(Gᵁ, Gⱽ, kernel_parameters)
end

@inline function augmented_kernel_size(grid::DistributedGrid)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    Tx, Ty, _ = topology(grid)

    Rx, Ry, _ = architecture(grid).ranks

    Ax = Rx == 1 ? Nx : (Tx == RightConnected || Tx == LeftConnected ? Nx + Hx - 1 : Nx + 2Hx - 2)
    Ay = Ry == 1 ? Ny : (Ty == RightConnected || Ty == LeftConnected ? Ny + Hy - 1 : Ny + 2Hy - 2)

    return (Ax, Ay)
end
   
@inline function augmented_kernel_offsets(grid::DistributedGrid)
    Hx, Hy, _ = halo_size(grid)
    Tx, Ty, _ = topology(grid)

    Rx, Ry, _ = architecture(grid).ranks

    Ax = Rx == 1 || Tx == RightConnected ? 0 : - Hx + 1
    Ay = Ry == 1 || Ty == RightConnected ? 0 : - Hy + 1

    return (Ax, Ay)
end

function FreeSurface(free_surface::SplitExplicitFreeSurface, velocities, grid::DistributedGrid)

        settings  = free_surface.settings 

        old_halos  = halo_size(grid)
        Nsubsteps  = length(settings.substepping.averaging_weights)

        new_halos = distributed_split_explicit_halos(old_halos, Nsubsteps+1, grid)         
        new_grid  = with_halo(new_halos, grid)

        η = ZFaceField(new_grid, indices = (:, :, size(new_grid, 3)+1))

        return SplitExplicitFreeSurface(η,
                                        SplitExplicitState(new_grid, settings.timestepper),
                                        SplitExplicitAuxiliaryFields(new_grid),
                                        free_surface.gravitational_acceleration,
                                        free_surface.settings)
end

@inline function distributed_split_explicit_halos(old_halos, step_halo, grid::DistributedGrid)

    Rx, Ry, _ = architecture(grid).ranks

    Ax = Rx == 1 ? old_halos[1] : step_halo
    Ay = Ry == 1 ? old_halos[2] : step_halo

    return (Ax, Ay, old_halos[3])
end

const DistributedSplitExplicit = SplitExplicitFreeSurface{<:DistributedField}

wait_free_surface_communication!(::DistributedSplitExplicit, ::SynchronizedDistributed) = nothing
    
function wait_free_surface_communication!(free_surface::DistributedSplitExplicit, arch)
    
    state = free_surface.state

    for field in (state.U̅, state.V̅)
        synchronize_communication!(field)
    end

    auxiliary = free_surface.auxiliary

    for field in (auxiliary.Gᵁ, auxiliary.Gⱽ)
        synchronize_communication!(field)
    end

    return nothing
end
