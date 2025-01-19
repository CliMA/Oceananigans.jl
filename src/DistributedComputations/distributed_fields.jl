using CUDA: CuArray
using OffsetArrays: OffsetArray
using Oceananigans.Grids: topology
using Oceananigans.Fields: validate_field_data, indices, validate_boundary_conditions
using Oceananigans.Fields: validate_indices, recv_from_buffers!, set_to_array!, set_to_field!

import Oceananigans.Fields: Field, FieldBoundaryBuffers, location, set!
import Oceananigans.BoundaryConditions: fill_halo_regions!

function Field((LX, LY, LZ)::Tuple, grid::DistributedGrid, data, old_bcs, indices::Tuple, op, status)
    indices = validate_indices(indices, (LX, LY, LZ), grid)
    validate_field_data((LX, LY, LZ), data, grid, indices)
    validate_boundary_conditions((LX, LY, LZ), grid, old_bcs)

    arch = architecture(grid)
    rank = arch.local_rank
    new_bcs = inject_halo_communication_boundary_conditions(old_bcs, rank, arch.connectivity, topology(grid))
    buffers = FieldBoundaryBuffers(grid, data, new_bcs)

    return Field{LX, LY, LZ}(grid, data, new_bcs, indices, op, status, buffers)
end

const DistributedField      = Field{<:Any, <:Any, <:Any, <:Any, <:DistributedGrid}
const DistributedFieldTuple = NamedTuple{S, <:NTuple{N, DistributedField}} where {S, N}

global_size(f::DistributedField) = global_size(architecture(f), size(f))

# Automatically partition under the hood if sizes are compatible
function set!(u::DistributedField, V::Union{Array, CuArray, OffsetArray})
    NV = size(V)
    Nu = global_size(u)

    # Suppress singleton indices
    NV′ = filter(n -> n > 1, NV)
    Nu′ = filter(n -> n > 1, Nu)

    if NV′ == Nu′
        v = partition(V, u)
    else
        v = V
    end

    return set_to_array!(u, v)
end

function set!(u::DistributedField, V::Field)
    if size(V) == global_size(u)
        v = partition(V, u)
        return set_to_array!(u, v)
    else
        return set_to_field!(u, V)
    end
end


"""
    synchronize_communication!(field)

complete the halo passing of `field` among processors.
"""
function synchronize_communication!(field)
    arch = architecture(field.grid)

    # Wait for outstanding requests
    if !isempty(arch.mpi_requests) 
        cooperative_waitall!(arch.mpi_requests)

        # Reset MPI tag
        arch.mpi_tag[] -= arch.mpi_tag[]

        # Reset MPI requests
        empty!(arch.mpi_requests)
    end

    recv_from_buffers!(field.data, field.boundary_buffers, field.grid)

    return nothing
end

# Fallback
reconstruct_global_field(field) = field

"""
    reconstruct_global_field(field::DistributedField)

Reconstruct a global field from a local field by combining the data from all processes.
"""
function reconstruct_global_field(field::DistributedField)
    global_grid = reconstruct_global_grid(field.grid)
    global_field = Field(location(field), global_grid)
    arch = architecture(field)

    global_data = construct_global_array(arch, interior(field), size(field))

    set!(global_field, global_data)

    return global_field
end
