using CUDA: @allowscalar
using Oceananigans.Utils: prettykeys

mutable struct NaNChecker{F, H}
    fields :: F
    hasnan :: H
    erroring :: Bool
end

default_nan_checker(model) = nothing

function Base.summary(nc::NaNChecker)
    fieldnames = prettykeys(nc.fields)
    if nc.erroring
        return "Erroring NaNChecker for $fieldnames"
    else
        return "NaNChecker for $fieldnames"
    end
end

Base.show(io, nc::NaNChecker) = print(io, summary(nc))

"""
    NaNChecker(; fields, erroring=false)

Return a `NaNChecker`, which sets `sim.running=false` if a `NaN` is detected
in any member of `fields` when `(::NaNChecker)(sim)` is called. `fields` should be
a container with key-value pairs like a dictionary or `NamedTuple`. 
`(::NaNChecker)(sim)` also returns a boolean indicating whether NaNs were found.

If `erroring=true`, the `NaNChecker` will throw an error on NaN detection.
"""
function NaNChecker(; fields, erroring=false)
    first_field = first(fields)
    hasnan = Field{Nothing, Nothing, Nothing}(first_field.grid, Bool)
    return NaNChecker(fields, hasnan, erroring)
end

function hasnan(field, checker)
    any!(isnan, checker.hasnan, field)
    return @allowscalar first(checker.hasnan)
end
    
hasnan(model::AbstractModel, checker) = hasnan(first(fields(model)), checker)

function (nc::NaNChecker)(simulation)
    found_nan = false
    for (name, field) in pairs(nc.fields)
        found_nan *= hasnan(field, nc)
        if found_nan
            simulation.running = false
            clock = simulation.model.clock
            t = time(simulation)
            iter = iteration(simulation)

            if nc.erroring
                error("time = $t, iteration = $iter: NaN found in field $name. Aborting simulation.")
            else
                @info "time = $t, iteration = $iter: NaN found in field $name. Stopping simulation."
            end
        end
    end
    return found_nan
end

"""
    erroring_NaNChecker!(simulation)

Toggle `simulation`'s `NaNChecker` to throw an error when a `NaN` is detected.
"""
function erroring_NaNChecker!(simulation)
    simulation.callbacks[:nan_checker].func.erroring = true
    return nothing
end
