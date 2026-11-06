module Fields

using ..Architectures: AnyConcreteReactantArray
using ..Grids: ReactantGrid, ShardedGrid
using ..OceananigansReactantExt: OceananigansReactantExt, deconcretize
using Oceananigans: Oceananigans
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Architectures: CPU, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: Field, ReducedAbstractField, copyable_fields, interior, interpolate!, set!
using Reactant

const ReactantField{LX, LY, LZ, O} = Field{LX, LY, LZ, O, <:ReactantGrid}
const ShardedDistributedField{LX, LY, LZ, O} = Field{LX, LY, LZ, O, <:ShardedGrid}
const ReactantOperation{LX, LY, LZ} = AbstractOperation{LX, LY, LZ, <:ReactantGrid}

Oceananigans.DistributedComputations.reconstruct_global_field(field::ShardedDistributedField) = field

#####
##### Indexing fields whose element type is traced
#####

# A field allocated inside a compiled function has a `TracedRNumber` element type. Indexing it is then
# ambiguous between the methods in `src/Fields/field.jl`, which specialize on the field's location
# (dropping the index along each reduced, `Nothing`, dimension) but take untyped indices, and Reactant's
# methods for arrays of traced numbers, which take `Int` or traced `Int` indices. We define one method
# for every combination of reduced dimensions, each more specific than both.

const TracedIndex = Union{Int, Reactant.TracedRNumber{Int}}

@inline reduced_index(::Type{Nothing}, i) = 1
@inline reduced_index(loc, i) = i

for LX in (:Nothing, :Any), LY in (:Nothing, :Any), LZ in (:Nothing, :Any)
    X = LX === :Nothing ? :Nothing : :(<:Any)
    Y = LY === :Nothing ? :Nothing : :(<:Any)
    Z = LZ === :Nothing ? :Nothing : :(<:Any)
    TracedField = :(Field{$X, $Y, $Z, <:Any, <:Any, <:Any, <:Any, <:Reactant.TracedRNumber})

    @eval begin
        @inline function Base.getindex(f::$TracedField, i::TracedIndex, j::TracedIndex, k::TracedIndex)
            LX, LY, LZ = Oceananigans.location(f)
            return getindex(f.data, reduced_index(LX, i), reduced_index(LY, j), reduced_index(LZ, k))
        end

        @inline function Base.setindex!(f::$TracedField, v, i::TracedIndex, j::TracedIndex, k::TracedIndex)
            LX, LY, LZ = Oceananigans.location(f)
            return setindex!(f.data, v, reduced_index(LX, i), reduced_index(LY, j), reduced_index(LZ, k))
        end
    end
end

OceananigansReactantExt.deconcretize(field::Field{LX, LY, LZ}) where {LX, LY, LZ} =
    Field{LX, LY, LZ}(field.grid,
                      deconcretize(field.data),
                      field.boundary_conditions,
                      field.indices,
                      field.operand,
                      field.status,
                      field.communication_buffers)


function Oceananigans.Fields.set_to_function!(u::ReactantField, f)
    # Supports serial and distributed
    arch = Oceananigans.Architectures.architecture(u)
    cpu_grid = on_architecture(CPU(), u.grid)
    cpu_u = Field(Oceananigans.Fields.instantiated_location(u), cpu_grid; indices=Oceananigans.Fields.indices(u))
    f_field = Oceananigans.Fields.field(Oceananigans.Fields.instantiated_location(u), f, cpu_grid)
    set!(cpu_u, f_field)
    interior(u) .= interior(cpu_u)
    return u
end

# When `v` may be copied into `u` we broadcast interiors, which traces. Otherwise we fall
# back to interpolation on the CPU, since interpolate!'s KA kernel does not
# currently trace under Reactant (see Reactant.jl#2364). This mirrors how
# set_to_function! hops to the CPU. Note that the CPU fallback only works
# outside of tracing: traced data cannot be materialized on the CPU mid-trace,
# so under `@compile`/`@jit` only the copy path is available.
#
# `copyable_fields` is Oceananigans' own copy-versus-interpolate policy; we differ from
# `copy_to_field!` only in mechanism, copying interiors alone rather than attempting halos
# through a `try`/`catch` that cannot be traced.
function Oceananigans.Fields.set_to_field!(u::ReactantField, v::ReactantField)
    if copyable_fields(u, v)
        interior(u) .= interior(v)
    else
        # As the generic method does, fill `v`'s halos with its own boundary conditions before interpolating
        fill_halo_regions!(v)
        cpu_grid_u = on_architecture(CPU(), u.grid)
        cpu_grid_v = on_architecture(CPU(), v.grid)
        cpu_u = Field(Oceananigans.Fields.instantiated_location(u), cpu_grid_u;
                      indices=Oceananigans.Fields.indices(u))
        cpu_v = Field(Oceananigans.Fields.instantiated_location(v), cpu_grid_v;
                      indices=Oceananigans.Fields.indices(v))
        # The whole parent, halos included: copying a view of a device array into a host array fails
        copyto!(parent(cpu_v), Array(parent(v)))
        interpolate!(cpu_u, cpu_v)
        copyto!(interior(u), interior(cpu_u))
    end
    return u
end

# Without this method, Reactant arrays reach the generic `u .= a` fallback, which indexes `a` with the
# field's own indices and so misreads windowed fields such as the free surface displacement.
# `TracedRArray` rather than `AnyTracedRArray`: the latter is `AbstractArray{<:TracedRNumber}`, which a
# `Field` of traced data (or an `OffsetArray` of one) also satisfies, making `set!` ambiguous with the
# generic `Field`-to-`Field` and array methods.
function Oceananigans.Fields.set!(u::ReactantField, a::Union{AnyConcreteReactantArray, Reactant.TracedRArray}, args...)
    copyto!(interior(u), a)
    return u
end

# Reactant reduces its own arrays natively, but has currently no path for a lazy `AbstractOperation`.
# We materialize the Field here: this isn't ideal but works until we have a better solution in Reactant
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
Oceananigans.DistributedComputations.synchronize_communication!(::ShardedDistributedField) = nothing

end
