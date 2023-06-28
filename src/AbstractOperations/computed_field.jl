#####
##### Fields computed from abstract operations
#####

using KernelAbstractions: @kernel, @index
using Oceananigans.Grids: default_indices
using Oceananigans.Fields: FieldStatus, reduced_dimensions, validate_indices, offset_index
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

    boundary_conditions = FieldBoundaryConditions(indices, boundary_conditions)

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
    @apply_regionally compute_at!(comp.operand, time)

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
