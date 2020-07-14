using Oceananigans.Architectures
using Oceananigans.BoundaryConditions
using Oceananigans.Utils
using Oceananigans.Grids: total_size

"""
    VolumeAverage{F, R, P, I, Ω, G} <: AbstractDiagnostic{F}

A diagnostic for computing the volume average of a field.
"""
mutable struct VolumeAverage{F, R, P, I, Ω, G} <: AbstractAverage{F}
          field :: F
         result :: P
      frequency :: Ω
       interval :: I
       previous :: Float64
    return_type :: R
           grid :: G
end

"""
    HorizontalAverage{F, R, P, I, Ω, G} <: AbstractDiagnostic{F}

A diagnostic for computing horizontal average of a field.
"""
mutable struct HorizontalAverage{F, R, P, I, Ω, G} <: AbstractAverage{F}
          field :: F
         result :: P
      frequency :: Ω
       interval :: I
       previous :: Float64
    return_type :: R
           grid :: G
end

"""
    ZonalAverage{F, R, P, I, Ω} <: AbstractDiagnostic

A diagnostic for computing the zonal average of a field.
"""
mutable struct ZonalAverage{F, R, P, I, Ω, G} <: AbstractAverage{F}
          field :: F
         result :: P
      frequency :: Ω
       interval :: I
       previous :: Float64
    return_type :: R
           grid :: G
end

"""
    VolumeAverage(model, field; frequency=nothing, interval=nothing, return_type=Array)

Construct a `VolumeAverage` of `field`.

After the volume average is computed it will be stored in the `result` property.

The `VolumeAverage` can be used as a callable object that computes and returns the
volume average.

A `frequency` or `interval` (or both) can be passed to indicate how often to run this
diagnostic if it is part of `model.diagnostics`. `frequency` is a number of iterations
while `interval` is a time interval in units of `model.clock.time`.

A `return_type` can be used to specify the type returned when the `VolumeAverage` is
used as a callable object. The default `return_type=Array` is useful when running a GPU
model and you want to save the output to disk by passing it to an output writer.
"""
function VolumeAverage(field; frequency=nothing, interval=nothing, return_type=Array)
    arch = architecture(field)
    result = zeros(arch, field.grid, 1, 1, 1)
    return VolumeAverage(field, result, frequency, interval, 0.0, return_type, field.grid)
end

"""
    HorizontalAverage(model, field; frequency=nothing, interval=nothing, return_type=Array)

Construct a `HorizontalAverage` of `field`.

After the horizontal average is computed it will be stored in the `result` property.

The `HorizontalAverage` can be used as a callable object that computes and returns the
horizontal average.

A `frequency` or `interval` (or both) can be passed to indicate how often to run this
diagnostic if it is part of `model.diagnostics`. `frequency` is a number of iterations
while `interval` is a time interval in units of `model.clock.time`.

A `return_type` can be used to specify the type returned when the `HorizontalAverage` is
used as a callable object. The default `return_type=Array` is useful when running a GPU
model and you want to save the output to disk by passing it to an output writer.
"""
function HorizontalAverage(field; frequency=nothing, interval=nothing, return_type=Array)
    arch = architecture(field)
    result = zeros(arch, field.grid, 1, 1, total_size(parent(field))[3])
    return HorizontalAverage(field, result, frequency, interval, 0.0, return_type, field.grid)
end

"""
    ZonalAverage(model, field; frequency=nothing, interval=nothing, return_type=Array)

Construct a `ZonalAverage` of `field`.

After the zonal average is computed it will be stored in the `result` property.

The `ZonalAverage` can be used as a callable object that computes and returns the
zonal average.

A `frequency` or `interval` (or both) can be passed to indicate how often to run this
diagnostic if it is part of `model.diagnostics`. `frequency` is a number of iterations
while `interval` is a time interval in units of `model.clock.time`.

A `return_type` can be used to specify the type returned when the `ZonalAverage` is
used as a callable object. The default `return_type=Array` is useful when running a GPU
model and you want to save the output to disk by passing it to an output writer.
"""
function ZonalAverage(field; frequency=nothing, interval=nothing, return_type=Array)
    arch = architecture(field)
    result = zeros(arch, field.grid, 1, total_size(parent(field))[2], total_size(parent(field))[3])
    return ZonalAverage(field, result, frequency, interval, 0.0, return_type, field.grid)
end

# Normalize sum to get the average
normalize_sum!(avg::VolumeAverage, grid) = avg.result ./= (grid.Nx * grid.Ny * grid.Nz)

normalize_sum!(avg::HorizontalAverage, grid) = avg.result ./= (grid.Nx * grid.Ny)

normalize_sum!(avg::ZonalAverage, grid) = avg.result ./= grid.Nx

"""
    run_diagnostic(model, avg::HorizontalAverage{NTuple{1}})

Compute the horizontal average of `avg.field` and store the result in `avg.result`.
"""
function run_diagnostic(model, avg::AbstractAverage)
    zero_halo_regions!(parent(avg.field), model.grid)
    sum!(avg.result, parent(avg.field))
    normalize_sum!(avg, model.grid)
    return nothing
end

function (avg::AbstractAverage)(model)
    run_diagnostic(model, avg)
    return avg.return_type(avg.result)
end
