module Fields

using Reactant

using Oceananigans: Oceananigans
using Oceananigans.Architectures: on_architecture, CPU
using Oceananigans.Fields: Field, interior

using KernelAbstractions: @index, @kernel

import Oceananigans.Fields: set_to_field!, set_to_function!, set!

import ..OceananigansReactantExt: deconcretize
import ..Grids: ReactantGrid
import ..Grids: ShardedGrid

const ReactantField{LX, LY, LZ, O} = Field{LX, LY, LZ, O, <:ReactantGrid}
const ShardedDistributedField{LX, LY, LZ, O} = Field{LX, LY, LZ, O, <:ShardedGrid}

deconcretize(field::Field{LX, LY, LZ}) where {LX, LY, LZ} =
    Field{LX, LY, LZ}(field.grid,
                      deconcretize(field.data),
                      field.boundary_conditions,
                      field.indices,
                      field.operand,
                      field.status,
                      field.communication_buffers)

function set!(u::ShardedDistributedField, V::ShardedDistributedField)
    @jit _set_to_field!(u, V)
    return nothing
end

function _set_to_field!(u, v)
    arch = Oceananigans.Architectures.architecture(u)
    Oceananigans.Utils.launch!(arch, u.grid, size(u), _copy!, u.data, v.data)
    return nothing
end

"""Compute an `operand` and store in `data`."""
@kernel function _copy!(u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds u[i, j, k] = v[i, j, k]
end

function set_to_function!(u::ReactantField, f)
    # Supports serial and distributed
    arch = Oceananigans.Architectures.architecture(u)
    cpu_grid = on_architecture(CPU(), u.grid)
    cpu_u = Field(Oceananigans.Fields.location(u), cpu_grid; indices=Oceananigans.Fields.indices(u))
    f_field = Oceananigans.Fields.field(Oceananigans.Fields.location(u), f, cpu_grid)
    set!(cpu_u, f_field)
    copyto!(parent(u), parent(cpu_u))
    return nothing
end

# keepin it simple
set_to_field!(u::ReactantField, v::ReactantField) = @jit _set_to_field!(u, v)

function set_to_function!(u::ShardedDistributedField, f)
    grid = u.grid
    arch = grid.architecture
    Oceananigans.Utils.launch!(arch, grid, size(u), _set_to_function_on_device!,
                               u, f, grid, Oceananigans.Fields.location(u))
    return nothing
end

@kernel function _set_to_function_on_device!(u, f, grid, loc)
    i, j, k = @index(Global, NTuple)
    LX, LY, LZ = loc
    x = Oceananigans.Grids.node(i, j, k, grid, LX(), LY(), LZ())
    @inbounds u[i, j, k] = f(x...)
end

end
