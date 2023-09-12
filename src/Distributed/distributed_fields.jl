import Oceananigans.Fields: Field, FieldBoundaryBuffers, location, set!
import Oceananigans.BoundaryConditions: fill_halo_regions!

using Oceananigans.Fields: validate_field_data, validate_boundary_conditions, validate_indices

function Field((LX, LY, LZ)::Tuple, grid::DistributedGrid, data, old_bcs, indices::Tuple, op, status)
    arch = architecture(grid)
    indices = validate_indices(indices, (LX, LY, LZ), grid)
    validate_field_data((LX, LY, LZ), data, grid, indices)
    validate_boundary_conditions((LX, LY, LZ), grid, old_bcs)
    new_bcs = inject_halo_communication_boundary_conditions(old_bcs, arch.local_rank, arch.connectivity)
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