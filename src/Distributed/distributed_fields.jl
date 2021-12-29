import Oceananigans.Fields: Field

function Field((LX, LY, LZ)::Tuple, grid::DistributedGrid, data, bcs, op, status)
    arch = architecture(grid)
    boundary_conditions = inject_halo_communication_boundary_conditions(bcs, arch.local_rank, arch.connectivity)
    return Field{LX, LY, LZ}(grid, data, bcs, op, status)
end
