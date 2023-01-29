import Oceananigans.Fields: Field, FieldBoundaryBuffers, location
import Oceananigans.BoundaryConditions: fill_halo_regions!

import Oceananigans.Grids: architecture

using Oceananigans.Fields: validate_field_data, validate_boundary_conditions, validate_indices

function Field((LX, LY, LZ)::Tuple, grid::DistributedGrid, data, old_bcs, indices::Tuple, op, status)
    arch = architecture(grid)
    validate_field_data((LX, LY, LZ), data, grid, indices)
    validate_boundary_conditions((LX, LY, LZ), grid, old_bcs)
    indices = validate_indices(indices, (LX, LY, LZ), grid)
    new_bcs = inject_halo_communication_boundary_conditions(old_bcs, arch.local_rank, arch.connectivity)
    buffers = FieldBoundaryBuffers(nothing, nothing, nothing, nothing)
    return Field{LX, LY, LZ}(grid, data, new_bcs, indices, op, status, buffers)
end

const DistributedField      = Field{<:Any, <:Any, <:Any, <:Any, <:DistributedGrid}
const DistributedFieldTuple = NamedTuple{S, <:NTuple{N, DistributedField}} where {S, N}

# To fix???
architecture(f::DistributedField) = child_architecture(architecture(f.grid))