module Fields

using Reactant

using Oceananigans: Oceananigans
using Oceananigans.Architectures: on_architecture, CPU
using Oceananigans.Fields: Field, interior, interpolate!

import Oceananigans.Fields: set_to_field!, set_to_function!, set!
import Oceananigans.DistributedComputations: reconstruct_global_field, synchronize_communication!

import ..OceananigansReactantExt: deconcretize
import ..Grids: ReactantGrid
import ..Grids: ShardedGrid

const ReactantField{LX, LY, LZ, O} = Field{LX, LY, LZ, O, <:ReactantGrid}
const ShardedDistributedField{LX, LY, LZ, O} = Field{LX, LY, LZ, O, <:ShardedGrid}

reconstruct_global_field(field::ShardedDistributedField) = field

deconcretize(field::Field{LX, LY, LZ}) where {LX, LY, LZ} =
    Field{LX, LY, LZ}(field.grid,
                      deconcretize(field.data),
                      field.boundary_conditions,
                      field.indices,
                      field.operand,
                      field.status,
                      field.communication_buffers)


function set_to_function!(u::ReactantField, f)
    # Supports serial and distributed
    arch = Oceananigans.Architectures.architecture(u)
    cpu_grid = on_architecture(CPU(), u.grid)
    cpu_u = Field(Oceananigans.Fields.instantiated_location(u), cpu_grid; indices=Oceananigans.Fields.indices(u))
    f_field = Oceananigans.Fields.field(Oceananigans.Fields.instantiated_location(u), f, cpu_grid)
    set!(cpu_u, f_field)
    copyto!(interior(u), interior(cpu_u))
    return nothing
end

# When sizes and locations match we can just copy interiors. Otherwise we fall
# back to interpolation on the CPU, since interpolate!'s KA kernel does not
# currently trace under Reactant (see Reactant.jl#2364). This mirrors how
# set_to_function! hops to the CPU.
function set_to_field!(u::ReactantField, v::ReactantField)
    if size(u) == size(v) && Oceananigans.location(u) == Oceananigans.location(v)
        interior(u) .= interior(v)
    else
        cpu_grid_u = on_architecture(CPU(), u.grid)
        cpu_grid_v = on_architecture(CPU(), v.grid)
        cpu_u = Field(Oceananigans.Fields.instantiated_location(u), cpu_grid_u;
                      indices=Oceananigans.Fields.indices(u))
        cpu_v = Field(Oceananigans.Fields.instantiated_location(v), cpu_grid_v;
                      indices=Oceananigans.Fields.indices(v))
        copyto!(interior(cpu_v), interior(v))
        interpolate!(cpu_u, cpu_v)
        copyto!(interior(u), interior(cpu_u))
    end
    return u
end

# No need to synchronize -> it should be implicit
synchronize_communication!(::ShardedDistributedField) = nothing

end
