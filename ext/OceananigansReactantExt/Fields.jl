module Fields

using Reactant

using Oceananigans: Oceananigans
using Oceananigans.Architectures: on_architecture, CPU
using Oceananigans.Fields: Field, interior

using KernelAbstractions: @index, @kernel

import Oceananigans.Fields: set_to_field!

import ..OceananigansReactantExt: deconcretize
import ..Grids: ReactantGrid

const ReactantField{LX, LY, LZ, O} = Field{LX, LY, LZ, O, <:ReactantGrid}

deconcretize(field::Field{LX, LY, LZ}) where {LX, LY, LZ} =
    Field{LX, LY, LZ}(field.grid,
                      deconcretize(field.data),
                      field.boundary_conditions,
                      field.indices,
                      field.operand,
                      field.status,
                      field.boundary_buffers)

# keepin it simple
set_to_field!(u::ReactantField, v::ReactantField) = @jit _set_to_field!(u, v)

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

end
