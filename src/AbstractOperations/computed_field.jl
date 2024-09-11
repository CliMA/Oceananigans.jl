#####
##### Fields computed from abstract operations
#####

using KernelAbstractions: @kernel, @index
using Oceananigans.Grids: default_indices
using Oceananigans.Fields: FunctionField, FieldStatus, reduced_dimensions, validate_indices, offset_index
using Oceananigans.Utils: launch!

import Oceananigans.Fields: Field, compute!

const OperationOrFunctionField = Union{AbstractOperation, FunctionField}
const ComputedField = Field{<:Any, <:Any, <:Any, <:OperationOrFunctionField}

"""
    Field(operand::OperationOrFunctionField;
          data = nothing,
          indices = indices(operand),
          boundary_conditions = FieldBoundaryConditions(operand.grid, location(operand)),
          recompute_safely = true)

Return a field `f` where `f.data` is computed from `f.operand` by calling `compute!(f)`.

Keyword arguments
=================

`data` (`AbstractArray`): An offset Array or CuArray for storing the result of a computation.
                          Must have `total_size(location(operand), grid)`.

`boundary_conditions` (`FieldBoundaryConditions`): Boundary conditions for `f`. 

`recompute_safely` (`Bool`): whether or not to _always_ "recompute" `f` if `f` is
                             nested within another computation via an `AbstractOperation` or `FunctionField`.
                             If `data` is not provided then `recompute_safely=false` and
                             recomputation is _avoided_. If `data` is provided, then
                             `recompute_safely = true` by default.
"""
function Field(operand::OperationOrFunctionField;
               data = nothing,
               indices = indices(operand),
               boundary_conditions = FieldBoundaryConditions(operand.grid, location(operand)),
               recompute_safely = true)

    grid = operand.grid
    loc = location(operand)
    indices = validate_indices(indices, loc, grid)

    @apply_regionally boundary_conditions = FieldBoundaryConditions(indices, boundary_conditions)

    if isnothing(data)
        data = new_data(grid, loc, indices)
        recompute_safely = false
    end

    status = recompute_safely ? nothing : FieldStatus()

    return Field(loc, grid, data, boundary_conditions, indices, operand, status)
end

"""
    compute!(comp::ComputedField)

Compute `comp.operand` and store the result in `comp.data`.
"""
function compute!(comp::ComputedField, time=nothing)
    # First compute `dependencies`:
    compute_at!(comp.operand, time)

    # Now perform the primary computation
    @apply_regionally compute_computed_field!(comp)

    fill_halo_regions!(comp)

    return comp
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

# struct FusedComputedFields{G, I}
#     fields :: Vector{ComputedField}
#     grid :: G
#     indices :: I

#     function FusedComputedFields(fields::Vector{ComputedField})
#         grid = first(fields).grid
#         indices = first(fields).indices
#         arch = architecture(first(fields))
#         sz = size(first(fields))

#         for field in fields
#             if !(field isa FullField) || (field isa ReducedField)
#                 throw(ArgumentError("All fields in FusedComputedFields must be FullField!"))
#             end
#             if field.grid !== grid
#                 throw(ArgumentError("All fields in FusedComputedFields must have the same grid!"))
#             end
#             if field.indices != indices
#                 throw(ArgumentError("All fields in FusedComputedFields must have the same indices!"))
#             end
#             if architecture(field) !== arch
#                 throw(ArgumentError("All fields in FusedComputedFields must have the same architecture!"))
#             end
#             if size(field) != sz
#                 throw(ArgumentError("All fields in FusedComputedFields must have the same size!"))
#             end
#         end

#         new(fields, grid, indices)
#     end
# end

function compute!(comps::Tuple{Vararg{ComputedField}}, time=nothing)
    grid = first(comps).grid
    indices = first(comps).indices
    arch = architecture(first(comps))
    sz = size(first(comps))

    # ordinary_field_ids = []

    for (i, comp) in enumerate(comps)
        comp.grid !== grid || throw(ArgumentError("All fields to compute must have the same grid!"))
        comp.indices != indices || throw(ArgumentError("All fields to compute must have the same indices!"))
        architecture(comp) !== arch || throw(ArgumentError("All fields to compute must have the same architecture!"))
        size(comp) != sz || throw(ArgumentError("All fields to compute must have the same size!"))
        (comp isa FullField) && !(comp isa ReducedField) || throw(ArgumentError("All fields to compute must be FullField!"))
    end

    for comp in comps
        compute_at!(comp.operand, time)
    end

    datas = Tuple(comp.data for comp in comps)
    operands = Tuple(comp.operand for comp in comps)

    @apply_regionally launch!(arch, grid, sz, _fused_compute!, datas, operands, indices)

    tupled_fill_halo_regions!(comps, grid)
end

@kernel function _fused_compute!(datas, operands, index_ranges)
    i, j, k = @index(Global, NTuple)

    for (data, operand) in zip(datas, operands)
        i′ = offset_compute_index(index_ranges[1], i)
        j′ = offset_compute_index(index_ranges[2], j)
        k′ = offset_compute_index(index_ranges[3], k)

        @inbounds data[i′, j′, k′] = operand[i′, j′, k′]
    end
end
