using Oceananigans.Models: AbstractModel

struct NaNChecker{F}
    fields :: F
end

"""
    NaNChecker(; fields)

Return a `NaNChecker`, which sets `sim.running=false` if a `NaN` is detected
in any member of `fields` when `NaNChecker(sim)` is called. `fields` should be
a container with key-value pairs like a dictionary or `NamedTuple`.
"""
NaNChecker(; fields) = NaNChecker(fields)

hasnan(field) = any(isnan, parent(field)) 
hasnan(model::AbstractModel) = hasnan(first(fields(model)))

function (nc::NaNChecker)(simulation)
    for (name, field) in pairs(nc.fields)
        if hasnan(field)
            @info "NaN found in field $name. Stopping simulation."
            simulation.running = false
        end
    end
    return nothing
end

