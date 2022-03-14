#####
##### Fields computed from abstract operations
#####

using KernelAbstractions: @kernel, @index
using Oceananigans.Grids: default_indices
using Oceananigans.Fields: FieldStatus, reduced_dimensions
using Oceananigans.Utils: launch!

import Oceananigans.Fields: Field, compute!

const ComputedField = Field{<:Any, <:Any, <:Any, <:AbstractOperation}

"""
    Field(operand::AbstractOperation; kwargs...)

Return `f::Field` where `f.data` is computed from `f.operand` by
calling compute!(f).

Keyword arguments
=================

data (AbstractArray): An offset Array or CuArray for storing the result of a computation.
                      Must have `total_size(location(operand), grid)`.

boundary_conditions (FieldBoundaryConditions): Boundary conditions for `f`. 

recompute_safely (Bool): whether or not to _always_ "recompute" `f` if `f` is
                         nested within another computation via an `AbstractOperation`.
                         If `data` is not provided then `recompute_safely=false` and
                         recomputation is _avoided_. If `data` is provided, then
                         `recompute_safely=true` by default.
"""
function Field(operand::AbstractOperation;
               data = nothing,
               indices = default_indices(3),
               boundary_conditions = FieldBoundaryConditions(operand.grid, location(operand)),
               recompute_safely = true)

    grid = operand.grid

    if isnothing(data)
        data = new_data(grid, location(operand), indices)
        recompute_safely = false
    end

    status = recompute_safely ? nothing : FieldStatus()

    return Field(location(operand), grid, data, boundary_conditions, indices, operand, status)
end

"""
    compute!(comp::ComputedField)

Compute `comp.operand` and store the result in `comp.data`.
"""
function compute!(comp::ComputedField, time=nothing)
    # First compute `dependencies`:
    compute_at!(comp.operand, time)

    arch = architecture(comp)
    event = launch!(arch, comp.grid, size(comp), _compute!, comp.data, comp.operand, comp.indices)
    wait(device(arch), event)

    fill_halo_regions!(comp)

    return comp
end

@inline offset_compute_index(::Colon, i) = i
@inline offset_compute_index(range::UnitRange, i) = range[1] + i - 1

"""Compute an `operand` and store in `data`."""
@kernel function _compute!(data, operand, index_ranges)
    i, j, k = @index(Global, NTuple)

    i′ = offset_compute_index(index_ranges[1], i)
    j′ = offset_compute_index(index_ranges[2], j)
    k′ = offset_compute_index(index_ranges[3], k)

    @inbounds data[i′, j′, k′] = operand[i′, j′, k′]
end

