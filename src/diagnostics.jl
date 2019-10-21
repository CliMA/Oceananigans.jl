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

"""
    HorizontalAverage{F, R, P, I, Ω} <: AbstractDiagnostic

A diagnostic for computing horizontal average of a field or the product of multiple fields.
"""
mutable struct HorizontalAverage{F, R, P, I, Ω} <: AbstractDiagnostic
        profile :: P
         fields :: F
      frequency :: Ω
       interval :: I
       previous :: Float64
    return_type :: R
end

"""
    HorizontalAverage(model, fields; frequency=nothing, interval=nothing, return_type=Array)

Construct a `HorizontalAverage` diagnostic for `model`.

After the horizontal average is computed it will be stored in the `profile` property.

The `HorizontalAverage` can be used as a callable object that computes and returns the
horizontal average.

If a single field is passed to `fields` the the horizontal average of that single field
will be computed. If multiple fields are passed to `fields`, then the horizontal average
of their product will be computed.

A `frequency` or `interval` (or both) can be passed to indicate how often to run this
diagnostic if it is part of `model.diagnostics`. `frequency` is a number of iterations
while `interval` is a time interval in units of `model.clock.time`.

A `return_type` can be used to specify the type returned when the `HorizontalAverage` is
used as a callable object. The default `return_type=Array` is useful when running a GPU
model and you want to save the output to disk by passing it to an output writer.

Warning
=======
Right now taking products of multiple fields does not take into account their locations
on the staggered grid and no attempt is made to interpolate all the different fields onto
a common location before calculating the product.
"""
function HorizontalAverage(model, fields; frequency=nothing, interval=nothing, return_type=Array)
    fields = parenttuple(tupleit(fields))
    profile = zeros(model.architecture, model.grid, 1, 1, model.grid.Tz)
    return HorizontalAverage(profile, fields, frequency, interval, 0.0, return_type)
end

# Normalize a horizontal sum to get the horizontal average.
normalize_horizontal_sum!(hsum, grid) = hsum.profile /= (grid.Nx * grid.Ny)

"""
    run_diagnostic(model, havg::HorizontalAverage{NTuple{1}})

Compute the horizontal average of `havg.fields` and store the result in `havg.profile`.
If length(fields) > 1, compute the product of the elements of fields (without taking
into account the possibility that they may have different locations in the staggered
grid) before computing the horizontal average.
"""
function run_diagnostic(model, havg::HorizontalAverage{NTuple{1}})
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

"""
    NaNChecker{F} <: AbstractDiagnostic

A diagnostic that checks for `NaN` values and aborts the simulation if any are found.
"""
struct NaNChecker{F} <: AbstractDiagnostic
    frequency :: Int
       fields :: F
end

"""
    NaNChecker(model; frequency, fields)

Construct a `NaNChecker` for `model`. `fields` should be a `Dict{Symbol,Field}`. A
`frequency` should be passed to indicate how often to check for NaNs (in number of
iterations).
"""
function NaNChecker(model; frequency, fields)
    return NaNChecker(frequency, fields)
end

function run_diagnostic(model::Model, nc::NaNChecker)
    for (name, field) in nc.fields
        if any(isnan, field.data.parent)
            t, i = model.clock.time, model.clock.iteration
            error("time = $t, iteration = $i: NaN found in $name. Aborting simulation.")
        end
    end
end

####
#### Timeseries diagnostics
####

"""
    Timeseries{D, Ω, I, T, TT} <: AbstractDiagnostic

A diagnostic for collecting and storing timeseries.
"""
struct Timeseries{D, Ω, I, T, TT} <: AbstractDiagnostic
    diagnostic :: D
     frequency :: Ω
      interval :: I
          data :: T
          time :: Vector{TT}
end

"""
    Timeseries(diagnostic, model; frequency=nothing, interval=nothing)

A `Timeseries` `Diagnostic` that records a time series of `diagnostic(model)`.

Example
=======
```julia
julia> model = BasicModel(N=(16, 16, 16), L=(1, 1, 1));

julia> max_u = Timeseries(FieldMaximum(abs, model.velocities.u), model; frequency=1)

julia> model.diagnostics[:max_u] = max_u; data(model.velocities.u) .= π; time_step!(model, Nt=3, Δt=1e-16)

julia> max_u.data
3-element Array{Float64,1}:
 3.141592653589793
 3.1415926025389127
 3.1415925323439517
```
"""
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

"""
    Timeseries(diagnostics::NamedTuple, model; frequency=nothing, interval=nothing)

A `Timeseries` `Diagnostic` that records a `NamedTuple` of time series of
`diag(model)` for each `diag` in `diagnostics`.

Example
=======
```julia
julia> model = BasicModel(N=(16, 16, 16), L=(1, 1, 1)); Δt = 1.0;

julia> cfl = Timeseries((adv=AdvectiveCFL(Δt), diff=DiffusiveCFL(Δt)), model; frequency=1);

julia> model.diagnostics[:cfl] = cfl; time_step!(model, Nt=3, Δt=Δt)

julia> cfl.data
(adv = [0.0, 0.0, 0.0, 0.0], diff = [0.0002688, 0.0002688, 0.0002688, 0.0002688])

julia> cfl.adv
4-element Array{Float64,1}:
 0.0
 0.0
 0.0
 0.0
```
"""
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

Examples
=======
```julia
julia> model = BasicModel(N=(16, 16, 16), L=(1, 1, 1));

julia> max_abs_u = FieldMaximum(abs, model.velocities.u);

julia> max_w² = FieldMaximum(x->x^2, model.velocities.w);
```
"""
struct FieldMaximum{F, M}
    mapping :: M
      field :: F
end

(m::FieldMaximum)(args...) = maximum(m.mapping, m.field.data.parent)

(m::FieldMaximum{<:NamedTuple})(args...) =
    NamedTuple{propertynames(m.field)}(maximum(m.mapping, f.data.parent) for f in m.field)

"""
    CFL{D, S}

An object for computing the Courant-Freidrichs-Lewy (CFL) number.
"""
struct CFL{D, S}
           Δt :: D
    timescale :: S
end

"""
    CFL(Δt [, timescale=Oceananigans.cell_advection_timescale])

Returns an object for computing the Courant-Freidrichs-Lewy (CFL) number
associated with time step or `TimeStepWizard` `Δt` and `timescale`.

See also `AdvectiveCFL` and `DiffusiveCFL`
"""
CFL(Δt) = CFL(Δt, cell_advection_timescale)

(c::CFL{<:Number})(model) = c.Δt / c.timescale(model)
(c::CFL{<:TimeStepWizard})(model) = c.Δt.Δt / c.timescale(model)

"""
    AdvectiveCFL(Δt)

Returns an object for computing the Courant-Freidrichs-Lewy (CFL) number
associated with time step or `TimeStepWizard` `Δt` and the time scale
for advection across a cell.

Example
=======
```julia
julia> model = BasicModel(N=(16, 16, 16), L=(8, 8, 8));

julia> cfl = AdvectiveCFL(1.0);

julia> data(model.velocities.u) .= π;

julia> cfl(model)
6.283185307179586
```
"""
AdvectiveCFL(Δt) = CFL(Δt, cell_advection_timescale)

"""
    DiffusiveCFL(Δt)

Returns an object for computing the Courant-Freidrichs-Lewy (CFL) number
associated with time step or `TimeStepWizard` `Δt` and the time scale
for diffusion across a cell associated with `model.closure`.

Example
=======
```julia
julia> model = BasicModel(N=(16, 16, 16), L=(1, 1, 1));

julia> cfl = DiffusiveCFL(0.1);

julia> cfl(model)
2.688e-5
```
"""
DiffusiveCFL(Δt) = CFL(Δt, cell_diffusion_timescale)
