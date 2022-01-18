#####
##### Fields computed from abstract operations
#####

using KernelAbstractions: @kernel, @index
using Oceananigans.Fields: FieldStatus, show_status, reduced_dimensions
using Oceananigans.Utils: launch!

import Oceananigans: short_show
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
               boundary_conditions = FieldBoundaryConditions(operand.grid, location(operand)),
               recompute_safely = true)

    grid = operand.grid

    if isnothing(data)
        data = new_data(grid, location(operand))
        recompute_safely = false
    end

    status = recompute_safely ? nothing : FieldStatus()

    return Field(location(operand), grid, data, boundary_conditions, operand, status)
end

"""
    compute!(comp::ComputedField)

Compute `comp.operand` and store the result in `comp.data`.
"""
function compute!(comp::ComputedField, time=nothing)
    # First compute `dependencies`:
    compute_at!(comp.operand, time)

    arch = architecture(comp)
    
    event = launch!(arch, comp.grid, :xyz, _compute!, comp.data, comp.operand;
                    include_right_boundaries = true,
                    location = location(comp),
                    reduced_dimensions = reduced_dimensions(comp))

    wait(device(arch), event)

    fill_halo_regions!(comp, arch)

    return comp
end

"""Compute an `operand` and store in `data`."""
@kernel function _compute!(data, operand)
    i, j, k = @index(Global, NTuple)
    @inbounds data[i, j, k] = operand[i, j, k]
end

short_show(field::ComputedField) = string("Field located at ", show_location(field), " computed from ", short_show(field.operand))

Base.show(io::IO, field::ComputedField) =
    print(io, "$(short_show(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field))\n",
          "├── grid: $(short_show(field.grid))\n",
          "├── operand: $(short_show(field.operand))\n",
          "└── status: $(show_status(field.status))")

