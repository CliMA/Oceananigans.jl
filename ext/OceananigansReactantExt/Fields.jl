module Fields

using Oceananigans.Architectures: on_architecture, CPU
using Oceananigans.Fields: Field

import ..OceananigansReactantExt: deconcretize
import ..Grids: ReactantGrid

deconcretize(field::Field{LX, LY, LZ}) where {LX, LY, LZ} =
    Field{LX, LY, LZ}(field.grid,
                      deconcretize(field.data),
                      field.boundary_conditions,
                      field.indices,
                      field.operand,
                      field.status,
                      field.boundary_buffers)


const ReactantField = Field{<:Any,
                            <:Any,
                            <:Any,
                            <:Any,
                            <:ReactantGrid}

copyto!(dest::ReactantField, src) = copyto!(interior(dest), src)

end
