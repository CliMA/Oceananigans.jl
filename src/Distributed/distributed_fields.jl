import Oceananigans.Fields: Field

function Field((LX, LY, LZ)::Tuple, grid::DistributedGrid, data, old_bcs, op, status)
    arch = architecture(grid)
    new_bcs = inject_halo_communication_boundary_conditions(old_bcs, arch.local_rank, arch.connectivity)
    return Field{LX, LY, LZ}(grid, data, new_bcs, op, status)
end
