using Adapt
using KernelAbstractions
using Oceananigans.Fields: AbstractField, FieldStatus, validate_field_data, new_data, datatuple, architecture
using Oceananigans.Architectures: device
using Oceananigans.Utils: work_layout

import Oceananigans.Fields: compute!

"""
    struct BuoyancyField{B, A, G, T} <: AbstractField{X, Y, Z, A, G}

Type representing buoyancy computed on the model grid.
"""
struct BuoyancyField{B, S, A, G, T} <: AbstractField{Cell, Cell, Cell, A, G}
        data :: A
        grid :: G
    buoyancy :: B
     tracers :: T
      status :: S

    """
        BuoyancyField(data, grid, buoyancy, tracers)
    
    Returns a `BuoyancyField` with `data` on `grid` corresponding to
    `buoyancy` computed from `tracers`.
    """
    function BuoyancyField(data, grid, buoyancy, tracers)

        validate_field_data(Cell, Cell, Cell, data, grid)

        status = recompute_safely ? FieldStatus(0.0) : nothing

        return new{typeof(buoyancy), typeof(status), typeof(data),
                   typeof(grid), typeof(tracers)}(data, grid, buoyancy, tracers, status)
    end
end

"""
    BuoyancyField(model; data=nothing)

Returns a `BuoyancyField` corresponding to `model.buoyancy`.
Calling `compute!(b::BuoyancyField)` computes the current buoyancy field
associated with `model` and stores the result in `b.data`.
"""
BuoyancyField(model; data=nothing) = _buoyancy_field(model.buoyancy, model.tracers, model.architecture, model.grid; data=data)

# Convenience for buoyancy=nothing
_buoyancy_field(::Nothing, args...; kwargs...) = nothing

#####
##### BuoyancyTracer
#####

_buoyancy_field(buoyancy::BuoyancyTracer, tracers, arch, grid; kwargs...) =
    BuoyancyField(tracers.b.data, grid, buoyancy, tracers)

compute!(::BuoyancyField{<:BuoyancyTracer}) = nothing
 
#####
##### Other buoyancy types
#####

function _buoyancy_field(buoyancy::AbstractBuoyancy, tracers, arch, grid; data=nothing)

    if isnothing(data)
        data = new_data(arch, grid, (Cell, Cell, Cell))
    end

    return BuoyancyField(data, grid, buoyancy, tracers)
end

"""
    compute!(buoyancy_field::BuoyancyField)

Compute the current `buoyancy_field` associated with `buoyancy_field.tracers` and store
the result in `buoyancy_field.data`.
"""
function compute!(buoyancy_field::BuoyancyField)

    data = buoyancy_field.data
    grid = buoyancy_field.grid
    tracers = datatuple(buoyancy_field.tracers)
    arch = architecture(data)

    workgroup, worksize = work_layout(grid, :xyz)

    compute_kernel! = compute_buoyancy!(device(arch), workgroup, worksize) 

    event = compute_kernel!(data, grid, buoyancy_field.buoyancy, tracers; dependencies=Event(device(arch)))

    wait(device(arch), event)

    return nothing
end

compute!(b::BuoyancyField{B, <:FieldStatus}, time) where B =
    conditional_compute!(b, time)

"""Compute an `operation` and store in `data`."""
@kernel function compute_buoyancy!(data, grid, buoyancy, C)
    i, j, k = @index(Global, NTuple)
    @inbounds data[i, j, k] = buoyancy_perturbation(i, j, k, grid, buoyancy, C)
end

#####
##### Adapt
#####

Adapt.adapt_structure(to, buoyancy_field::BuoyancyField) =
    BuoyancyField(Adapt.adapt(to, buoyancy_field.data),
                  Adapt.adapt(to, buoyancy_field.grid),
                  Adapt.adapt(to, buoyancy_field.buoyancy),
                  Adapt.adapt(to, buoyancy_field.tracers))
