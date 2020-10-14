"""
    NaNChecker{F} <: AbstractDiagnostic

A diagnostic that checks for `NaN` values and aborts the simulation if any are found.
"""
struct NaNChecker{T, F} <: AbstractDiagnostic
    trigger :: T
     fields :: F
end

"""
    NaNChecker(; trigger, fields)

Returns a `NaNChecker` that checks for `NaN` anywhere within `fields`
when `trigger` actuates.
"""
NaNChecker(; trigger, fields) = NaNChecker(trigger, fields)

function run_diagnostic!(nc::NaNChecker, model)
    for (name, field) in nc.fields
        if any(isnan, field.data.parent)
            t, i = model.clock.time, model.clock.iteration
            error("time = $t, iteration = $i: NaN found in $name. Aborting simulation.")
        end
    end
end
