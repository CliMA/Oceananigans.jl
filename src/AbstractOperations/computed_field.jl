#####
##### Fields computed from abstract operations
#####

using KernelAbstractions: @kernel, @index
using Oceananigans.Grids: Center, Face, default_indices, new_data
using Oceananigans.Fields: FunctionField, FieldBoundaryConditions, FieldStatus, validate_indices,
    offset_index, instantiated_location, set_status!, uses_quadfolded_vector_boundary_conditions
using Oceananigans.Utils: KernelParameters, launch!, @apply_regionally

import Oceananigans.Fields: Field, compute!

const OperationOrFunctionField = Union{AbstractOperation, FunctionField}
const ComputedField = Field{<:Any, <:Any, <:Any, <:OperationOrFunctionField}

"""
    Field(operand::OperationOrFunctionField;
          data = nothing,
          indices = indices(operand),
          boundary_conditions = FieldBoundaryConditions(operand.grid, location(operand)),
          compute = true,
          recompute_safely = true)

Return a field `f` where `f.data` is computed from `f.operand` by calling `compute!(f)`.

Keyword arguments
=================

`data` (`AbstractArray`): An offset Array or CuArray for storing the result of a computation.
                          Must have `total_size(location(operand), grid)`.

`boundary_conditions` (`FieldBoundaryConditions`): Boundary conditions for `f`.

`recompute_safely` (`Bool`): Whether or not to _always_ "recompute" `f` if `f` is
                             nested within another computation via an `AbstractOperation` or `FunctionField`.
                             If `data` is not provided then `recompute_safely = false` and
                             recomputation is _avoided_. If `data` is provided, then
                             `recompute_safely = true` by default.

`compute`: If `true`, `compute!` the `Field` during construction, otherwise if `false`, initialize with zeros.
           Default: `true`.
"""
function Field(operand::OperationOrFunctionField;
               data = nothing,
               indices = indices(operand),
               boundary_conditions = FieldBoundaryConditions(operand.grid, instantiated_location(operand)),
               status = nothing,
               compute = true,
               recompute_safely = true)

    grid = operand.grid
    loc = instantiated_location(operand)
    indices = validate_indices(indices, loc, grid)

    @apply_regionally boundary_conditions = FieldBoundaryConditions(indices, boundary_conditions)

    if isnothing(data)
        @apply_regionally data = new_data(eltype(operand), grid, loc, indices)
        recompute_safely = false
    end

    if isnothing(status)
        status = recompute_safely ? nothing : FieldStatus()
    end

    computed_field = Field(loc, grid, data, boundary_conditions, indices, operand, status)

    if compute
        compute!(computed_field)
    end

    return computed_field
end

"""
    compute!(comp::ComputedField, time=nothing)

Compute `comp.operand` and store the result in `comp.data`.
If `time` then only compute dependency fields with `time != field.status.time`.
"""
function compute!(comp::ComputedField, time=nothing)
    # First compute `dependencies`:
    compute_at!(comp.operand, time)

    # Now perform the primary computation
    @apply_regionally compute_computed_field!(comp)
    fill_computed_field_halos!(comp)

    # Update status
    set_status!(comp.status, time)

    return comp
end

function compute!(collection::Tuple{<:ComputedField{Face, Center, LZ},
                                    <:ComputedField{Center, Face, LZ}},
                  time=nothing) where LZ
    u, v = collection

    compute_at!(u.operand, time)
    compute_at!(v.operand, time)

    @apply_regionally begin
        compute_computed_field!(u)
        compute_computed_field!(v)
    end

    fill_halo_regions!((u, v))
    set_status!(u.status, time)
    set_status!(v.status, time)

    return collection
end

@inline paired_horizontal_computed_fields(::ComputedField{Face, Center, LZ},
                                          ::ComputedField{Center, Face, LZ}) where LZ = true
@inline paired_horizontal_computed_fields(u, v) = false

@inline function paired_horizontal_computed_field_name_pairs(collection::NamedTuple)
    field_names = collect(keys(collection))
    used_pair = falses(length(field_names))
    uv_pairs = Tuple{Symbol, Symbol}[]

    for u_index in eachindex(field_names)
        used_pair[u_index] && continue

        u_name = field_names[u_index]
        u = getproperty(collection, u_name)

        u isa ComputedField{Face, Center} || continue

        for v_index in (u_index + 1):length(field_names)
            used_pair[v_index] && continue

            v_name = field_names[v_index]
            v = getproperty(collection, v_name)

            if v isa ComputedField{Center, Face} &&
               paired_horizontal_computed_fields(u, v)
                push!(uv_pairs, (u_name, v_name))
                used_pair[u_index] = true
                used_pair[v_index] = true
                break
            end
        end
    end

    return Tuple(uv_pairs)
end

@inline function paired_horizontal_computed_field_index_pairs(collection::Tuple)
    used_pair = falses(length(collection))
    uv_pairs = NTuple{2, Int}[]

    for u_index in eachindex(collection)
        used_pair[u_index] && continue

        u = collection[u_index]
        u isa ComputedField{Face, Center} || continue

        for v_index in (u_index + 1):length(collection)
            used_pair[v_index] && continue

            v = collection[v_index]

            if v isa ComputedField{Center, Face} &&
               paired_horizontal_computed_fields(u, v)
                push!(uv_pairs, (u_index, v_index))
                used_pair[u_index] = true
                used_pair[v_index] = true
                break
            end
        end
    end

    return Tuple(uv_pairs)
end

function compute!(collection::NamedTuple, time=nothing)
    uv_pairs = paired_horizontal_computed_field_name_pairs(collection)

    if !isempty(uv_pairs)
        for (u_name, v_name) in uv_pairs
            compute!((getproperty(collection, u_name), getproperty(collection, v_name)), time)
        end

        for name in keys(collection)
            pair_computed = any(name === u_name || name === v_name for (u_name, v_name) in uv_pairs)
            pair_computed && continue

            compute!(getproperty(collection, name), time)
        end

        return collection
    end

    map(field -> compute!(field, time), collection)
    return collection
end

function compute!(collection::Tuple, time=nothing)
    uv_pairs = paired_horizontal_computed_field_index_pairs(collection)

    if !isempty(uv_pairs)
        for (u_index, v_index) in uv_pairs
            compute!((collection[u_index], collection[v_index]), time)
        end

        for field_index in eachindex(collection)
            pair_computed = any(field_index == u_index || field_index == v_index for (u_index, v_index) in uv_pairs)
            pair_computed && continue

            compute!(collection[field_index], time)
        end

        return collection
    end

    map(field -> compute!(field, time), collection)
    return collection
end

function compute!(collection::NamedTuple{(:u, :v),
                                         Tuple{<:ComputedField{Face, Center, LZ},
                                               <:ComputedField{Center, Face, LZ}}},
                  time=nothing) where LZ
    compute!((collection.u, collection.v), time)
    return collection
end

@inline function fill_computed_field_halos!(comp::ComputedField)
    if uses_quadfolded_vector_boundary_conditions(comp)
        return nothing
    end

    fill_halo_regions!(comp)
    return nothing
end

function compute_computed_field!(comp)
    arch = architecture(comp)
    parameters = KernelParameters(size(comp), map(offset_index, comp.indices))
    launch!(arch, comp.grid, parameters, _compute!, comp.data, comp.operand)
    return comp
end

"""Compute an `operand` and store in `data`."""
@kernel function _compute!(data, operand)
    i, j, k = @index(Global, NTuple)
    @inbounds data[i, j, k] = operand[i, j, k]
end
