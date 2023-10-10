import Oceananigans.Fields: Field, FieldBoundaryBuffers, location, set!
import Oceananigans.BoundaryConditions: fill_halo_regions!

using Oceananigans.Grids: topology
using Oceananigans.Fields: validate_field_data, indices, validate_boundary_conditions, validate_indices, recv_from_buffers!

function Field((LX, LY, LZ)::Tuple, grid::DistributedGrid, data, old_bcs, indices::Tuple, op, status)
    arch = architecture(grid)
    indices = validate_indices(indices, (LX, LY, LZ), grid)
    validate_field_data((LX, LY, LZ), data, grid, indices)
    validate_boundary_conditions((LX, LY, LZ), grid, old_bcs)
    new_bcs = inject_halo_communication_boundary_conditions(old_bcs, arch.local_rank, arch.connectivity, topology(grid))
    buffers = FieldBoundaryBuffers(grid, data, new_bcs)

    return Field{LX, LY, LZ}(grid, data, new_bcs, indices, op, status, buffers)
end

const DistributedField      = Field{<:Any, <:Any, <:Any, <:Any, <:DistributedGrid}
const DistributedFieldTuple = NamedTuple{S, <:NTuple{N, DistributedField}} where {S, N}

function set!(u::DistributedField, f::Function)
    arch = architecture(u)
    if child_architecture(arch) isa GPU
        cpu_grid = on_architecture(cpu_architecture(arch), u.grid)
        u_cpu = Field(location(u), cpu_grid; indices = indices(u))
        f_field = field(location(u), f, cpu_grid)
        set!(u_cpu, f_field)
        set!(u, u_cpu)
    elseif child_architecture(arch) isa CPU
        f_field = field(location(u), f, u.grid)
        set!(u, f_field)
    end

    return u
end

# Automatically partition under the hood if sizes are compatible
function set!(u::DistributedField, v::AbstractArray)
    gsize = global_size(architecture(u), size(u))

    if size(v) == size(u)
        f = arch_array(architecture(u), f)
        u .= f
        return u
    elseif size(v) == gsize
        f = partition_global_array(architecture(u), v, size(u))
        u .= f
        return u
    else
        throw(ArgumentError("ERROR: DimensionMismatch: array could not be set to match destination field"))
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
