#####
##### Fields computed from abstract operations
#####

using KernelAbstractions: @kernel, @index
using Oceananigans.Grids: default_indices
using Oceananigans.Fields: FieldStatus, reduced_dimensions, validate_indices, offset_compute_index
using Oceananigans.Utils: launch!

import Oceananigans.Fields: Field, compute!

const ComputedField = Field{<:Any, <:Any, <:Any, <:AbstractOperation}

"""
    Field(operand::AbstractOperation;
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
                             nested within another computation via an `AbstractOperation`.
                             If `data` is not provided then `recompute_safely=false` and
                             recomputation is _avoided_. If `data` is provided, then
                             `recompute_safely = true` by default.
"""
function Field(operand::AbstractOperation;
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
    launch!(arch, comp.grid, size(comp), _compute!, comp.data, comp.operand, comp.indices)
    return comp
end

"""Compute an `operand` and store in `data`."""
@kernel function _compute!(data, operand, index_ranges)
    i, j, k = @index(Global, NTuple)

    i′ = offset_compute_index(index_ranges[1], i)
    j′ = offset_compute_index(index_ranges[2], j)
    k′ = offset_compute_index(index_ranges[3], k)

    @inbounds data[i′, j′, k′] = operand[i′, j′, k′]
end

struct FusedComputedFields
    fields::Vector{ComputedField}
    grid::AbstractGrid
    datas::Vector{AbstractArray}
    operands::Vector{AbstractOperation}
    indices::Tuple

    function FusedComputedFields(fields::Vector{ComputedField})
        grid = fields[1].grid
        indices = fields[1].indices
        arch = architecture(fields[1])
        sz = size(fields[1])
        for field in fields[2:end]
            @assert field.grid === grid
            @assert field.indices == indices
            @assert architecture(field) === arch
            @assert size(field) == sz
        end
        datas = [field.data for field in fields]
        operands = [field.operand for field in fields]
        new(fields, grid, datas, operands, indices)
    end
end

function compute!(fused::FusedComputedFields, time=nothing)
    # First compute `dependencies`:
    for operand in fused.operands
        compute_at!(operand, time)
    end

    # Now perform the primary computation
    @apply_regionally compute_fused_computed_fields!(fused)

    for comp in fused.fields
        fill_halo_regions!(comp)
    end

    return fused
end

function compute_fused_computed_fields!(fused)
    arch = architecture(fused.fields[1])
    launch!(arch, fused.grid, size(fused.fields[1]), _fused_compute!, 
            fused.data, fused.operands, fused.indices)
    return fused
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
