####
#### Useful kernels
####

function velocity_div!(grid::RegularCartesianGrid, u, v, w, div)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds div[i, j, k] = div_f2c(grid, u, v, w, i, j, k)
            end
        end
    end
end

####
#### Horizontally averaged vertical profiles
####

mutable struct HorizontalAverage{F, R, P, I, Ω} <: Diagnostic
        profile :: P
         fields :: F
      frequency :: Ω
       interval :: I
       previous :: Float64
    return_type :: R
end


function HorizontalAverage(model, fields; frequency=nothing, interval=nothing, return_type=nothing)
    fields = parenttuple(tupleit(fields))
    profile = zeros(model.arch, model.grid, 1, 1, model.grid.Tz)
    return HorizontalAverage(profile, fields, frequency, interval, 0.0, return_type)
end

"Normalize a horizontal sum to get the horizontal average."
normalize_horizontal_sum!(hsum, grid) = hsum.profile /= (grid.Nx * grid.Ny)

"""
    run_diagnostic(model, havg)

Compute the horizontal average of `havg.fields` and store the
result in `havg.profile`. If length(fields) > 1, compute the
product of the elements of fields (without taking into account
the possibility that they may have different locations in the
staggered grid) before computing the horizontal average.
"""
function run_diagnostic(model::Model, havg::HorizontalAverage{NTuple{1}})
    zero_halo_regions!(havg.fields[1], model.grid)
    sum!(havg.profile, havg.fields[1])
    normalize_horizontal_sum!(havg, model.grid)
    return
end

function run_diagnostic(model::Model, havg::HorizontalAverage)
    zero_halo_regions!(havg.fields, model.grid)

    # Use pressure as scratch space for the product of fields.
    tmp = model.pressures.pNHS.data.parent
    zero_halo_regions!(tmp, model.grid)

    @. tmp = *(havg.fields...)
    sum!(havg.profile, tmp)
    normalize_horizontal_sum!(havg, model.grid)

    return
end

function (havg::HorizontalAverage{F, Nothing})(model) where F
    run_diagnostic(model, havg)
    return havg.profile
end

function (havg::HorizontalAverage)(model)
    run_diagnostic(model, havg)
    return havg.return_type(havg.profile)
end

####
#### NaN checker
####

struct NaNChecker{D} <: Diagnostic
    frequency :: Int
       fields :: D
end

function NaNChecker(model; frequency=1000, fields=Dict(:w => model.velocities.w.data.parent))
    return NaNChecker(frequency, fields)
end

function run_diagnostic(model::Model, nc::NaNChecker)
    for (name, field) in nc.fields
        if any(isnan, field)
            t, i = model.clock.time, model.clock.iteration
            error("time = $t, iteration = $i: NaN found in $name. Aborting simulation.")
        end
    end
end

####
#### Timeseries diagnostics
####

function run_diagnostic(model, diag::AbstractTimeseriesDiagnostic) 
    push!(diag.data, diag(model))
    push!(diag.time, model.clock.time)
    return nothing
end

"""
    TimeseriesDiagnostic(calculate, model; frequency=nothing, interval=nothing)

A generic AbstractTimeseriesDiagnostic that records a timeseries of `calculate(model)`.
"""
struct TimeseriesDiagnostic{Ω, I, C, TD, TT}
    calculate :: C
    frequency :: Ω
     interval :: I
         data :: Vector{TD}
         time :: Vector{TT}
end

function TimeseriesDiagnostic(calculate, model; frequency=nothing, interval=nothing)
    TT = typeof(model.clock.time)
    TD = typeof(calculate(model))
    return TimeseriesDiagnostic(calculate, frequency, interval, TD[], TT[])
end

@inline (timeseries::TimeseriesDiagnostic)(model) = timeseries.calculate(model)

"""
    MaxAbsFieldDiagnostic(field, frequency=nothing, interval=nothing) <: AbstractTimeseriesDiagnostic

A diagnostic for recording timeseries of the maximum absolute value of a field.
"""
struct MaxAbsFieldDiagnostic{Ω, I, T, F} <: AbstractTimeseriesDiagnostic
        field :: F
    frequency :: Ω
     interval :: I
         data :: Vector{T}
         time :: Vector{T}
end

function MaxAbsFieldDiagnostic(field; frequency=nothing, interval=nothing) 
    T = typeof(maximum(abs, field.data.parent))
    return MaxAbsFieldDiagnostic(field, frequency, interval, T[], T[]) 
end

(m::MaxAbsFieldDiagnostic)(model) = maximum(abs, m.field.data.parent)

"""
    CFLDiagnostic([T=Float64], Δt; frequency=nothing, interval=nothing, 
                  timescale=Oceananigans.cell_advection_timescale)

A diagnostic for computing timeseries of type `T` of the 
Courant-Freidrichs-Lewis (CFL) number, associated with time-step `Δt` 
and a characteristic `timescale`. If `Δt` is `TimeStepWizard` object, 
the current `Δt` associated with the wizard is used.

See also `AdvectiveCFLDiagnostic` and `DiffusiveCFLDiagnostic`.
"""
struct CFLDiagnostic{D, Ω, I, T, S} <: AbstractTimeseriesDiagnostic
           Δt :: D
    frequency :: Ω
     interval :: I
         data :: Vector{T}
         time :: Vector{T}
    timescale :: S
end

function CFLDiagnostic(T, Δt; frequency=nothing, interval=nothing, 
                       timescale=cell_advection_timescale)

    return CFLDiagnostic(Δt, frequency, interval, T[], T[], timescale)
end

CFLDiagnostic(Δt; kwargs...) = CFLDiagnostic(Float64, Δt; kwargs...)

(c::CFLDiagnostic{<:Number})(model) = c.Δt / c.timescale(model)
(c::CFLDiagnostic{<:TimeStepWizard})(model) = c.Δt.Δt / c.timescale(model)

"""
    AdvectiveCFLDiagnostic([T=Float64], Δt; frequency=nothing, interval=nothing)

A diagnostic for computing timeseries of the advective CFL number.
"""
AdvectiveCFLDiagnostic(T, Δt; kwargs...) = 
    CFLDiagnostic(T, Δt; timescale=cell_advection_timescale, kwargs...)

AdvectiveCFLDiagnostic(Δt; kwargs...) = 
    CFLDiagnostic(Float64, Δt; timescale=cell_advection_timescale, kwargs...)

"""
    DiffusiveCFLDiagnostic([T=Float64], Δt; frequency=nothing, interval=nothing)

A diagnostic for computing timeseries of the diffusive CFL number.
"""
DiffusiveCFLDiagnostic(T, Δt; kwargs...) = 
    CFLDiagnostic(T, Δt; timescale=cell_diffusion_timescale, kwargs...)

DiffusiveCFLDiagnostic(Δt; kwargs...) = 
    CFLDiagnostic(Float64, Δt; timescale=cell_diffusion_timescale, kwargs...)
