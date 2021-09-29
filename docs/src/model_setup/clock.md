# Clock

The clock holds the current simulation time, iteration number, and time step stage.
The time step stage is relevant only for the multi-stage time-stepper `RungeKutta3TimeStepper`.

By default, `Clock`s are initialized at iteration 0, and stage 1,

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```jldoctest
julia> clock = Clock(time=0.0)
Clock{Float64}: time = 0 seconds, iteration = 0, stage = 1
```

but can be modified to start the model clock at some other time.
For example, passing

```jldoctest
julia> clock = Clock(time=3600.0)
Clock{Float64}: time = 1 hour, iteration = 0, stage = 1
```

to the constructor for `NonhydrostaticModel` causes the simulation
time to start at ``t = 3600`` seconds.

The type of the keyword argument `time` should be a float or date type.
To use the date type `TimeDate` from the `TimesDates.jl` package,
for example, pass

```jldoctest
julia> using TimesDates

julia> clock = Clock(time=TimeDate(2020))
Clock{TimesDates.TimeDate}: time = 2020-01-01T00:00:00, iteration = 0, stage = 1
```

to `NonhydrostaticModel`.
`TimeDate` supports nanosecond resolution and is thus recommended over `Base.Dates.DateTime`,
which is also supported but has only millisecond resolution.
