# Clock

The clock holds the current iteration number and time. By default the model starts at iteration number 0 and time 0

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```jldoctest
julia> clock = Clock(0.0, 0)
Clock{Float64}: time = 0.000 s, iteration = 0
```

but can be modified if you wish to start the model clock at some other time. If you want iteration 0 to correspond to
$t = 3600$ seconds, then you can construct

```jldoctest
julia> clock = Clock(3600.0, 0)
Clock{Float64}: time = 1.000 hr, iteration = 0
```

and pass it to the model.
