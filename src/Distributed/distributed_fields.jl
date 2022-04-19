import Oceananigans.Fields: Field, location

function Field((LX, LY, LZ)::Tuple, grid::DistributedGrid, data, old_bcs, indices::Tuple, op, status)
    arch = architecture(grid)
    new_bcs = inject_halo_communication_boundary_conditions(old_bcs, arch.local_rank, arch.connectivity)
    return Field{LX, LY, LZ}(grid, data, new_bcs, indices, op, status)
end

const DistributedField      = Field{<:Any, <:Any, <:Any, <:Any, <:DistributedGrid}
const DistributedFieldTuple = NamedTuple{S, <:NTuple{N, DistributedField}} where {S, N}

