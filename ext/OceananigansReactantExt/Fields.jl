module Fields

using Reactant

using Oceananigans: Oceananigans
using Oceananigans.AbstractOperations: AbstractOperation, BinaryOperation, KernelFunctionOperation,
                                       evaluate_kernel_function_operation
using Oceananigans.Architectures: on_architecture, CPU
using Oceananigans.Fields: Field, ReducedAbstractField, interior, interpolate!

import Oceananigans.Fields: set_to_field!, set_to_function!, set!
import Oceananigans.DistributedComputations: reconstruct_global_field, synchronize_communication!

import ..OceananigansReactantExt: deconcretize
import ..Grids: ReactantGrid
import ..Grids: ShardedGrid

const ReactantField{LX, LY, LZ, O} = Field{LX, LY, LZ, O, <:ReactantGrid}
const ShardedDistributedField{LX, LY, LZ, O} = Field{LX, LY, LZ, O, <:ShardedGrid}
const ReactantOperation{LX, LY, LZ} = AbstractOperation{LX, LY, LZ, <:ReactantGrid}

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

# A dimension can be copied by broadcasting when both fields share its location and size,
# or when v is reduced there (Nothing location, singleton size): broadcasting expands the
# singleton across u's dimension, as when a 1D reference column is set into a 3D field.
@inline broadcastable_dimension(ℓu, ℓv, Nu, Nv) = (ℓu == ℓv && Nu == Nv) || (ℓv === Nothing && Nv == 1)

function broadcast_compatible(u, v)
    ℓu = Oceananigans.location(u)
    ℓv = Oceananigans.location(v)
    Nu = size(u)
    Nv = size(v)
    return all(broadcastable_dimension(ℓu[d], ℓv[d], Nu[d], Nv[d]) for d in 1:3)
end

# When v broadcasts against u we can just copy interiors. Otherwise we fall
# back to interpolation on the CPU, since interpolate!'s KA kernel does not
# currently trace under Reactant (see Reactant.jl#2364). This mirrors how
# set_to_function! hops to the CPU. Note that the CPU fallback only works
# outside of tracing: traced data cannot be materialized on the CPU mid-trace,
# so under `@compile`/`@jit` only the broadcast path is available.
function set_to_field!(u::ReactantField, v::ReactantField)
    if broadcast_compatible(u, v)
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

# `traced_type_inner` gives `BinaryOperation` and `KernelFunctionOperation` the eltype of their traced
# grid, which Reactant needs so that reductions route through `overloaded_mapreduce`. Since
# `AnyTracedRArray` is defined purely by eltype (`AbstractArray{TracedRNumber{T}, N}`), the operations
# then satisfy it and Reactant's indexing becomes ambiguous with the operations' own. Reactant's method
# assumes a view-like wrapper and reindexes into a single ancestor buffer, which an operation does not
# have, so resolve in favour of Oceananigans'. Without this, the `_compute!` kernel of a computed field
# fails to compile whenever the grid carries array-valued coordinates. These cannot be restricted to a
# `ReactantGrid`: inside a kernel the adapted grid has lost its architecture parameter.
const TracedIndex = Union{Int, Reactant.TracedRNumber{Int}}

@inline Base.getindex(β::BinaryOperation, i::TracedIndex, j::TracedIndex, k::TracedIndex) =
    β.op(i, j, k, β.grid, β.▶a, β.▶b, β.a, β.b)

@inline Base.getindex(κ::KernelFunctionOperation, i::TracedIndex, j::TracedIndex, k::TracedIndex) =
    evaluate_kernel_function_operation(κ, i, j, k)

# Reactant reduces its own arrays natively, but has no path for a lazy `AbstractOperation` as the
# reduction source: `mapreducedim!` delegates to `mapreduce`, whose fallback walks the operation
# element-by-element, which traced data rejects with "Scalar indexing is disallowed". Compute the
# operation into a `Field` first / materialize it.
for reduction in (:sum, :maximum, :minimum, :all, :any, :prod)

    reduction! = Symbol(reduction, '!')

    @eval begin
        Base.$(reduction!)(f::Function, r::ReducedAbstractField, a::ReactantOperation; kwargs...) =
            Base.$(reduction!)(f, r, Field(a); kwargs...)

        Base.$(reduction!)(r::ReducedAbstractField, a::ReactantOperation; kwargs...) =
            Base.$(reduction!)(r, Field(a); kwargs...)

        Base.$(reduction!)(f, r::AbstractArray, a::ReactantOperation; kwargs...) =
            Base.$(reduction!)(f, r, Field(a); kwargs...)

        Base.$(reduction)(f::Function, a::ReactantOperation; kwargs...) =
            Base.$(reduction)(f, Field(a); kwargs...)
    end
end

# No need to synchronize -> it should be implicit
synchronize_communication!(::ShardedDistributedField) = nothing

end
