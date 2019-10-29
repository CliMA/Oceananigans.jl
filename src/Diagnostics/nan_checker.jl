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

function run_diagnostic(model, nc::NaNChecker)
    for (name, field) in nc.fields
        if any(isnan, field.data.parent)
            t, i = model.clock.time, model.clock.iteration
            error("time = $t, iteration = $i: NaN found in $name. Aborting simulation.")
        end
    end
end
