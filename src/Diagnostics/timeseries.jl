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
julia> model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)));

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
julia> model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))); Δt = 1.0;

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
