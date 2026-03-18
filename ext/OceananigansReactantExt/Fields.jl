module Fields

using Reactant

using Oceananigans: Oceananigans
using Oceananigans.Architectures: on_architecture, CPU
using Oceananigans.Fields: Field, interior

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

# keepin it simple
set_to_field!(u::ReactantField, v::ReactantField) = interior(u) .= interior(v)

# No need to synchronize -> it should be implicit
synchronize_communication!(::ShardedDistributedField) = nothing

end
