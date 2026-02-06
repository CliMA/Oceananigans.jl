using Oceananigans.Grids: topology
using Oceananigans.Fields: validate_field_data, indices, validate_boundary_conditions, instantiated_location
using Oceananigans.Fields: validate_indices, set_to_array!, set_to_field!
using GPUArraysCore: @allowscalar

using Oceananigans.Fields: ReducedAbstractField,
                           get_neutral_mask,
                           condition_operand,
                           initialize_reduced_field!,
                           filltype,
                           reduced_dimensions,
                           reduced_location
using Oceananigans.Fields: condition_operand, conditional_length
using LinearAlgebra: dot, norm
using Statistics: mean

import Oceananigans.Fields: Field, set!
import Oceananigans.BoundaryConditions: fill_halo_regions!
import LinearAlgebra: norm, dot
import Statistics: mean

function Field(loc::Tuple{<:LX, <:LY, <:LZ}, grid::DistributedGrid, data, old_bcs, indices::Tuple, op, status) where {LX, LY, LZ}
    indices = validate_indices(indices, loc, grid)
    validate_field_data(loc, data, grid, indices)
    validate_boundary_conditions(loc, grid, old_bcs)

    arch = architecture(grid)
    rank = arch.local_rank
    new_bcs = inject_halo_communication_boundary_conditions(old_bcs, rank, arch.connectivity, topology(grid))
    buffers = communication_buffers(grid, data, new_bcs)

    return Field{LX, LY, LZ}(grid, data, new_bcs, indices, op, status, buffers)
end

const DistributedField         = Field{<:Any, <:Any, <:Any, <:Any, <:DistributedGrid}
const DistributedFieldTuple    = NamedTuple{S, <:NTuple{N, DistributedField}} where {S, N}
const DistributedAbstractField = AbstractField{<:Any, <:Any, <:Any, <:DistributedGrid}

global_size(f::DistributedField) = global_size(architecture(f), size(f))

# Automatically partition under the hood if sizes are compatible
set!(u::DistributedField, V::Union{Array, OffsetArray}) = _set!(u, V)

function _set!(u::DistributedField, V::VT) where {VT}
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

# Fallback -> not implemented
synchronize_communication!(var) = throw(ArgumentError("`synchronize_communication!` not implemented for variables of type $(typeof(var))"))

# Methods for types that do not require synchronization
synchronize_communication!(::AbstractField) = nothing
synchronize_communication!(::AbstractArray) = nothing
synchronize_communication!(::Number)        = nothing
synchronize_communication!(::Nothing)       = nothing

# Distribute synchronize_communication! over tuples and named tuples
synchronize_communication!(t::Union{NamedTuple, Tuple}) = foreach(synchronize_communication!, t)

"""
    synchronize_communication!(field)

complete the halo passing of `field` among processors.
"""
function synchronize_communication!(field::DistributedField)
    arch = architecture(field.grid)

    # Wait for outstanding requests
    if !isempty(arch.mpi_requests)
        cooperative_waitall!(arch.mpi_requests)

        # Reset MPI tag
        arch.mpi_tag[] = 0

        # Reset MPI requests
        empty!(arch.mpi_requests)
    end

    recv_from_buffers!(field.data, field.communication_buffers, field.grid)

    return nothing
end

# Fallback
reconstruct_global_field(field) = field

"""
    reconstruct_global_field(field::DistributedField)

Reconstruct a global field from a local field by combining the data from all processes.
"""
function reconstruct_global_field(field::DistributedField)
    arch = architecture(field)
    field_indices = field.indices

    if (!(field_indices[1] isa Colon) && (arch.ranks[1] != 1)) ||
       (!(field_indices[2] isa Colon) && (arch.ranks[2] != 1))
        @warn "Windowed fields in a partitioned directions are not supported."
    end

    global_grid = reconstruct_global_grid(field.grid)
    global_field = Field(instantiated_location(field), global_grid; indices=field_indices)

    global_data = construct_global_array(interior(field), arch, size(field))

    set!(global_field, global_data)

    return global_field
end

"""
    partition_dimensions(arch::Distributed)
    partition_dimensions(f::DistributedField)

Return the partitioned dimensions of a distributed field or architecture.
"""
function partition_dimensions(arch::Distributed)
    R = ranks(arch)
    dims = []
    for r in eachindex(R)
        if R[r] > 1
            push!(dims, r)
        end
    end
    return tuple(dims...)
end

partition_dimensions(f::DistributedField) = partition_dimensions(architecture(f))

function maybe_all_reduce!(op, f::ReducedAbstractField)
    reduced_dims   = reduced_dimensions(f)
    partition_dims = partition_dimensions(f)

    arch = architecture(f)
    sync_device!(arch)

    if any([dim ∈ partition_dims for dim in reduced_dims])
        all_reduce!(op, parent(f), architecture(f))
    end

    return f
end

# Allocating and in-place reductions
for (reduction, all_reduce_op) in zip((:sum, :maximum, :minimum, :all, :any, :prod),
                                      (:+,   :max,     :min,     :&,   :|,   :*))

    reduction! = Symbol(reduction, '!')

    @eval begin
        # In-place
        function Base.$(reduction!)(f::Function,
                                    r::ReducedAbstractField,
                                    a::DistributedAbstractField;
                                    condition = nothing,
                                    mask = get_neutral_mask(Base.$(reduction!)),
                                    kwargs...)

            operand = condition_operand(f, a, condition, mask)

            Base.$(reduction!)(identity,
                               interior(r),
                               operand;
                               kwargs...)

            return maybe_all_reduce!($(all_reduce_op), r)
        end

        function Base.$(reduction!)(r::ReducedAbstractField,
                                    a::DistributedAbstractField;
                                    condition = nothing,
                                    mask = get_neutral_mask(Base.$(reduction!)),
                                    kwargs...)

            Base.$(reduction!)(identity,
                               interior(r),
                               condition_operand(a, condition, mask);
                               kwargs...)

            return maybe_all_reduce!($(all_reduce_op), r)
        end

        # Allocating
        function Base.$(reduction)(f::Function,
                                   c::DistributedAbstractField;
                                   condition = nothing,
                                   mask = get_neutral_mask(Base.$(reduction!)),
                                   dims = :)

            conditioned_c = condition_operand(f, c, condition, mask)
            T = filltype(Base.$(reduction!), c)
            loc = reduced_location(instantiated_location(c); dims)
            r = Field(loc, c.grid, T; indices=indices(c))
            initialize_reduced_field!(Base.$(reduction!), identity, r, conditioned_c)
            Base.$(reduction!)(identity, interior(r), conditioned_c, init=false)

            maybe_all_reduce!($(all_reduce_op), r)

            if dims isa Colon
                return @allowscalar first(r)
            else
                return r
            end
        end
    end
end

# Distributed norm
@inline function norm(u::DistributedField; condition=nothing)
    n² = dot(u, u; condition)
    return sqrt(n²)
end

# Distributed dot product
@inline function dot(u::DistributedField, v::DistributedField; condition=nothing)
    cu = condition_operand(u, condition, 0)
    cv = condition_operand(v, condition, 0)

    B = cu * cv # Binary operation
    r = zeros(u.grid, 1)

    Base.mapreducedim!(identity, +, r, B)
    dot_local = @allowscalar r[1]
    arch = architecture(u)
    return all_reduce(+, dot_local, arch)
end

@inline function _mean(f, c::DistributedAbstractField, ::Colon; condition=nothing, mask=0)
    operand = condition_operand(f, c, condition, mask)

    global_sum = sum(operand)
    global_length = conditional_length(operand)

    return global_sum / global_length
end

@inline function _mean(f, c::DistributedAbstractField, dims; condition=nothing, mask=0)
    operand = condition_operand(f, c, condition, mask)
    r = sum(operand; dims)
    L = conditional_length(operand, dims)

    if L isa Field
        parent(r) ./= parent(L)
    else
        parent(r) ./= L
    end

    return r
end

@inline mean(f::Function, c::DistributedAbstractField; condition=nothing, dims=:) =
    _mean(f, c, dims; condition)

@inline mean(f::Function, c::DistributedAbstractField, dims; condition=nothing, mask=0) =
    _mean(f, c, dims; condition, mask)

@inline mean(c::DistributedAbstractField; condition=nothing, dims=:) = _mean(identity, c, dims; condition)
