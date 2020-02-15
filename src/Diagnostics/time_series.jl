"""
    TimeSeries{D, Ω, I, T, TT} <: AbstractDiagnostic

A diagnostic for collecting and storing time series.
"""
struct TimeSeries{D, Ω, I, T, TT} <: AbstractDiagnostic
    diagnostic :: D
     frequency :: Ω
      interval :: I
          data :: T
          time :: Vector{TT}
end

"""
    TimeSeries(diagnostic, model; frequency=nothing, interval=nothing)

A `TimeSeries` `Diagnostic` that records a time series of `diagnostic(model)`.

Example
=======
```julia
julia> model = IncompressibleModel(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)));

julia> max_u = TimeSeries(FieldMaximum(abs, model.velocities.u), model; frequency=1)

julia> model.diagnostics[:max_u] = max_u; data(model.velocities.u) .= π; time_step!(model, Nt=3, Δt=1e-16)

julia> max_u.data
3-element Array{Float64,1}:
 3.141592653589793
 3.1415926025389127
 3.1415925323439517
```
"""
function TimeSeries(diagnostic, model; frequency=nothing, interval=nothing)
    TD = typeof(diagnostic(model))
    TT = typeof(model.clock.time)
    return TimeSeries(diagnostic, frequency, interval, TD[], TT[])
end

@inline (time_series::TimeSeries)(model) = time_series.diagnostic(model)

function run_diagnostic(model, diag::TimeSeries)
    push!(diag.data, diag(model))
    push!(diag.time, model.clock.time)
    return nothing
end

"""
    TimeSeries(diagnostics::NamedTuple, model; frequency=nothing, interval=nothing)

A `TimeSeries` `Diagnostic` that records a `NamedTuple` of time series of
`diag(model)` for each `diag` in `diagnostics`.

Example
=======
```julia
julia> model = IncompressibleModel(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))); Δt = 1.0;

julia> cfl = TimeSeries((adv=AdvectiveCFL(Δt), diff=DiffusiveCFL(Δt)), model; frequency=1);

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
function TimeSeries(diagnostics::NamedTuple, model; frequency=nothing, interval=nothing)
    TT = typeof(model.clock.time)
    TDs = Tuple(typeof(diag(model)) for diag in diagnostics)
    data = NamedTuple{propertynames(diagnostics)}(Tuple(T[] for T in TDs))
    return TimeSeries(diagnostics, frequency, interval, data, TT[])
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
