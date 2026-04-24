using Oceananigans.Utils: prettykeys
using Oceananigans.Solvers: iteration

mutable struct NaNChecker{F}
    fields :: F
    erroring :: Bool
    nan_detected :: Bool
end

NaNChecker(fields) = NaNChecker(fields, false, false) # default
default_nan_checker(model) = nothing

function Base.summary(nc::NaNChecker)
    fieldnames = prettykeys(nc.fields)
    if nc.erroring
        return "Erroring NaNChecker for $fieldnames"
    else
        return "NaNChecker for $fieldnames"
    end
end

Base.show(io::IO, nc::NaNChecker) = print(io, summary(nc))

"""
    NaNChecker(; fields, erroring=false)

Return a `NaNChecker`, which sets `sim.running=false` if a `NaN` is detected
in any member of `fields` when `NaNChecker(sim)` is called. `fields` should be
a container with key-value pairs like a dictionary or `NamedTuple`.

If `erroring=true`, the `NaNChecker` will throw an error on NaN detection.
NaN detection state is tracked internally and can be queried with
`nan_detected(nan_checker)`.
"""
NaNChecker(; fields, erroring=false) = NaNChecker(fields, erroring, false)

# Allow `:nan_checker` callbacks that are not `NaNChecker` to integrate with
# simulation logic that queries and resets NaN detection state.
nan_detected(::Any) = false
nan_detected(nc::NaNChecker) = nc.nan_detected

reset_nan_checker!(::Any) = nothing
function reset_nan_checker!(nc::NaNChecker)
    nc.nan_detected = false
    return nothing
end

hasnan(field::AbstractArray) = any(isnan, parent(field))
hasnan(model) = hasnan(first(fields(model)))

function (nc::NaNChecker)(simulation)
    for (name, field) in pairs(nc.fields)
        if hasnan(field)
            simulation.running = false
            nc.nan_detected = true
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
    return nothing
end

"""
    erroring_NaNChecker!(simulation)

Toggle `simulation`'s `NaNChecker` to throw an error when a `NaN` is detected.
"""
function erroring_NaNChecker!(simulation)
    simulation.callbacks[:nan_checker].func.erroring = true
    return nothing
end
