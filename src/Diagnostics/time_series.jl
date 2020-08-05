"""
    TimeSeries{D, II, TI, T, TT} <: AbstractDiagnostic

A diagnostic for collecting and storing time series.
"""
struct TimeSeries{D, II, TI, T, TT} <: AbstractDiagnostic
            diagnostic :: D
    iteration_interval :: II
         time_interval :: TI
                  data :: T
                  time :: Vector{TT}
end

"""
    TimeSeries(diagnostic, model; iteration_interval=nothing, time_interval=nothing)

A `TimeSeries` `Diagnostic` that records a time series of `diagnostic(model)`.

Example
=======

```jldoctest timeseries1
julia> using Oceananigans, Oceananigans.Diagnostics

julia> model = IncompressibleModel(grid=RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1)));

julia> set!(model, u=π);

julia> sim = Simulation(model, Δt=1, stop_iteration=3);

julia> max_u = TimeSeries(FieldMaximum(abs, model.velocities.u), model; iteration_interval=1);

julia> sim.diagnostics[:max_u] = max_u;

julia> run!(sim);
[ Info: Simulation is stopping. Model iteration 3 has hit or exceeded simulation stop iteration 3.
```

```jldoctest timeseries1
julia> max_u.data
4-element Array{Float64,1}:
 3.141592653589793
 3.141592653589793
 3.141592653589793
 3.141592653589793
```
"""
function TimeSeries(diagnostic, model; iteration_interval=nothing, time_interval=nothing)
    TD = typeof(diagnostic(model))
    TT = typeof(model.clock.time)
    return TimeSeries(diagnostic, iteration_interval, time_interval, TD[], TT[])
end

@inline (time_series::TimeSeries)(model) = time_series.diagnostic(model)

function run_diagnostic(model, diag::TimeSeries)
    push!(diag.data, diag(model))
    push!(diag.time, model.clock.time)
    return nothing
end

"""
    TimeSeries(diagnostics::NamedTuple, model; iteration_interval=nothing, time_interval=nothing)

A `TimeSeries` `Diagnostic` that records a `NamedTuple` of time series of
`diag(model)` for each `diag` in `diagnostics`.

Example
=======

```jldoctest timeseries2
julia> using Oceananigans, Oceananigans.Diagnostics

julia> model = IncompressibleModel(grid=RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1)));

julia> Δt = 1.0;

julia> cfl = TimeSeries((adv=AdvectiveCFL(Δt), diff=DiffusiveCFL(Δt)), model; iteration_interval=1);

julia> sim = Simulation(model, Δt=Δt, stop_iteration=3);

julia> sim.diagnostics[:cfl] = cfl;

julia> run!(sim);
[ Info: Simulation is stopping. Model iteration 3 has hit or exceeded simulation stop iteration 3.
```

```jldoctest timeseries2
julia> cfl.data
(adv = [0.0, 0.0, 0.0, 0.0], diff = [0.0002688, 0.0002688, 0.0002688, 0.0002688])
```

``` jldoctest timeseries2
julia> cfl.diff
4-element Array{Float64,1}:
 0.0002688
 0.0002688
 0.0002688
 0.0002688
```
"""
function TimeSeries(diagnostics::NamedTuple, model; iteration_interval=nothing, time_interval=nothing)
    TT = typeof(model.clock.time)
    TDs = Tuple(typeof(diag(model)) for diag in diagnostics)
    data = NamedTuple{propertynames(diagnostics)}(Tuple(T[] for T in TDs))
    return TimeSeries(diagnostics, iteration_interval, time_interval, data, TT[])
end

function (time_series::TimeSeries{<:NamedTuple})(model)
    names = propertynames(time_series.diagnostic)
    return NamedTuple{names}(Tuple(diag(model) for diag in time_series.diagnostics))
end

function run_diagnostic(model, diag::TimeSeries{<:NamedTuple})
    ntuple(Val(length(diag.diagnostic))) do i
        Base.@_inline_meta
        push!(diag.data[i], diag.diagnostic[i](model))
    end
    push!(diag.time, model.clock.time)
    return nothing
end

Base.getproperty(ts::TimeSeries{<:NamedTuple}, name::Symbol) = get_time_series_field(ts, name)

"Returns a field of time series, or a field of `time_series.data` when possible."
function get_time_series_field(time_series, name)
    if name ∈ propertynames(time_series)
        return getfield(time_series, name)
    else
        return getproperty(time_series.data, name)
    end
end
