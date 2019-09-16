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

mutable struct HorizontalAverage{F, R, P, I, Ω} <: AbstractDiagnostic
        profile :: P
         fields :: F
      frequency :: Ω
       interval :: I
       previous :: Float64
    return_type :: R
end

function HorizontalAverage(model, fields; frequency=nothing, interval=nothing, return_type=Array)
    fields = parenttuple(tupleit(fields))
    profile = zeros(model.architecture, model.grid, 1, 1, model.grid.Tz)
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

struct NaNChecker{F} <: AbstractDiagnostic
    frequency :: Int
       fields :: F
end

function NaNChecker(model; frequency, fields)
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

"""
    Timeseries(diagnostic, model; frequency=nothing, interval=nothing)

A generic `Timeseries` `Diagnostic` that records a time series of `diagnostic(model)`.

Example
=======

cfl = Timeseries(CFL(Δt), model; frequency=1)
"""
struct Timeseries{D, Ω, I, T, TT} <: AbstractDiagnostic
    diagnostic :: D
     frequency :: Ω
      interval :: I
          data :: T
          time :: Vector{TT}
end

# Single-diagnostic timeseries
function Timeseries(diagnostic, model; frequency=nothing, interval=nothing)
    TD = typeof(diagnostic(model))
    TT = typeof(model.clock.time)
    return Timeseries(diagnostic, frequency, interval, TD[], TT[])
end

@inline (timeseries::Timeseries)(model) = timeseries.diagnostic(model)

function run_diagnostic(model, diag::Timeseries) 
    push!(diag.data, diag(model))
    push!(diag.time, model.clock.time)
    return nothing
end

# Multi-diagnostic timeseries
function Timeseries(diagnostics::NamedTuple, model; frequency=nothing, interval=nothing)
    TT = typeof(model.clock.time)
    TDs = Tuple(typeof(diag(model)) for diag in diagnostics)
    data = NamedTuple{propertynames(diagnostics)}(Tuple(T[] for T in TDs))
    return Timeseries(diagnostics, frequency, interval, data, TT[])
end

function (timeseries::Timeseries{<:NamedTuple})(model) 
    names = propertynames(timeseries.diagnostic)
    return NamedTuple{names}(Tuple(diag(model) for diag in timeseries.diagnostics))
end

function run_diagnostic(model, diag::Timeseries{<:NamedTuple}) 
    ntuple(Val(length(diag.diagnostic))) do i
        Base.@_inline_meta
        push!(diag.data[i], diag.diagnostic[i](model))
    end
    push!(diag.time, model.clock.time)
    return nothing
end

Base.getproperty(ts::Timeseries{<:NamedTuple}, name::Symbol) = get_timeseries_field(ts, name)

"Returns a field of timeseries, or a field of `timeseries.data` when possible."
function get_timeseries_field(timeseries, name) 
    if name ∈ propertynames(timeseries)
        return getfield(timeseries, name)
    else
        return getproperty(timeseries.data, name)
    end
end

"""
    FieldMaximum(mapping, field)

An object for calculating the maximum of a `mapping` function applied 
element-wise to `field`.

Example
=======

julia> max_abs_u = FieldMaximum(abs, model.velocities.u)

julia> max_w² = FieldMaximum(x->x^2, model.velocities.w)

julia> max_abs_u_timeseries = Timeseries(FieldMaximum(abs, model.velocities.u), model; frequency=1)

julia> max_abs_U_timeseries = Timeseries(FieldMaximum(abs, model.velocities), model; frequency=1)
"""
struct FieldMaximum{F, M}
    mapping :: M
      field :: F
end

(m::FieldMaximum)(args...) = maximum(m.mapping, m.field.data.parent)

(m::FieldMaximum{<:NamedTuple})(args...) = 
    NamedTuple{propertynames(m.field)}(maximum(m.mapping, f.data.parent) for f in m.field)

"""
    CFL(Δt, timescale=Oceananigans.cell_advection_timescale)

A diagnostic for computing the Courant-Freidrichs-Lewis (CFL) 
number, associated with time-step `Δt` and a characteristic `timescale`. 
If `Δt` is `TimeStepWizard` object, the current `Δt` associated with the 
wizard is used.

See also `AdvectiveCFL` and `DiffusiveCFL`.

Example
=======

cfl = CFL(0.5)
cfl(model)
"""
struct CFL{D, S}
           Δt :: D
    timescale :: S
end

CFL(Δt) = CFL(Δt, cell_advection_timescale)

(c::CFL{<:Number})(model) = c.Δt / c.timescale(model)
(c::CFL{<:TimeStepWizard})(model) = c.Δt.Δt / c.timescale(model)

"""
    AdvectiveCFLDiagnostic([T=Float64], Δt; frequency=nothing, interval=nothing)

A diagnostic for computing time series of the advective CFL number.
"""
AdvectiveCFL(Δt) = CFL(Δt, cell_advection_timescale)

"""
    DiffusiveCFLDiagnostic([T=Float64], Δt; frequency=nothing, interval=nothing)

A diagnostic for computing time series of the diffusive CFL number.
"""
DiffusiveCFL(Δt) = CFL(Δt, cell_diffusion_timescale)
