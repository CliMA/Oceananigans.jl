using Adapt
using KernelAbstractions
using Oceananigans.Fields: AbstractField, FieldStatus, validate_field_data, new_data, conditional_compute!
using Oceananigans.Fields: architecture, tracernames
using Oceananigans.Architectures: device
using Oceananigans.Utils: work_layout

import Oceananigans.Fields: compute!, compute_at!

import Oceananigans: short_show

"""
    struct BuoyancyField{B, A, G, T} <: AbstractField{X, Y, Z, A, G}

Type representing buoyancy computed on the model grid.
"""
struct BuoyancyField{B, S, A, G, T} <: AbstractField{Center, Center, Center, A, G}
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
    function BuoyancyField(data, grid, buoyancy, tracers, recompute_safely::Bool)

        validate_field_data(Center, Center, Center, data, grid)

        status = recompute_safely ? nothing : FieldStatus(zero(eltype(grid)))

        return new{typeof(buoyancy), typeof(status), typeof(data),
                   typeof(grid), typeof(tracers)}(data, grid, buoyancy, tracers, status)
    end

    function BuoyancyField(data, grid, buoyancy, tracers, status)
        validate_field_data(Center, Center, Center, data, grid)
        return new{typeof(buoyancy), typeof(status), typeof(data),
                   typeof(grid), typeof(tracers)}(data, grid, buoyancy, tracers, status)
    end
end

"""
    BuoyancyField(model; data=nothing, recompute_safely=true)

Returns a `BuoyancyField` corresponding to `model.buoyancy`.
Calling `compute!(b::BuoyancyField)` computes the current buoyancy field
associated with `model` and stores the result in `b.data`.
"""
BuoyancyField(model; data=nothing, recompute_safely=true) =
    _buoyancy_field(model.buoyancy, model.tracers, model.architecture, model.grid, data, recompute_safely)

# Convenience for buoyancy=nothing
_buoyancy_field(::Nothing, args...; kwargs...) = nothing

#####
##### BuoyancyTracer
#####

_buoyancy_field(buoyancy::BuoyancyTracer, tracers, arch, grid, args...) =
    BuoyancyField(tracers.b.data, grid, buoyancy, tracers, true)

compute!(::BuoyancyField{<:BuoyancyTracer}, time=nothing) = nothing
 
#####
##### Other buoyancy types
#####

function _buoyancy_field(buoyancy::AbstractBuoyancy, tracers, arch, grid,
                         data, recompute_safely)

    if isnothing(data)
        data = new_data(arch, grid, (Center, Center, Center))
        recompute_safely = false
    end

    return BuoyancyField(data, grid, buoyancy, tracers, recompute_safely)
end

"""
    compute!(buoyancy_field::BuoyancyField)

Compute the current `buoyancy_field` associated with `buoyancy_field.tracers` and store
the result in `buoyancy_field.data`.
"""
function compute!(buoyancy_field::BuoyancyField, time=nothing)

    data = buoyancy_field.data
    grid = buoyancy_field.grid
    tracers = buoyancy_field.tracers
    buoyancy = buoyancy_field.buoyancy
    arch = architecture(data)

    workgroup, worksize = work_layout(grid, :xyz)

    compute_kernel! = compute_buoyancy!(device(arch), workgroup, worksize) 

    event = compute_kernel!(data, grid, buoyancy, tracers; dependencies=Event(device(arch)))

    wait(device(arch), event)

    return nothing
end

compute_at!(b::BuoyancyField{B, <:FieldStatus}, time) where B =
    conditional_compute!(b, time)

"""Compute an `operation` and store in `data`."""
@kernel function compute_buoyancy!(data, grid, buoyancy, tracers)
    i, j, k = @index(Global, NTuple)
    @inbounds data[i, j, k] = buoyancy_perturbation(i, j, k, grid, buoyancy, tracers)
end

#####
##### Adapt
#####

Adapt.adapt_structure(to, buoyancy_field::BuoyancyField) = Adapt.adapt(to, buoyancy_field.data)

#####
##### Show
#####

short_show(field::BuoyancyField) = string("BuoyancyField for ", typeof(field.buoyancy))

show(io::IO, field::BuoyancyField) =
    print(io, "$(short_show(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field.data))\n",
          "├── grid: $(short_show(field.grid))", '\n',
          "├── buoyancy: $(typeof(field.buoyancy))", '\n',
          "├── tracers: $(tracernames(field.tracers))", '\n',
          "└── status: ", show_status(field.status), '\n')
