struct NaNChecker{T, F} <: AbstractDiagnostic
    schedule :: T
      fields :: F
end

"""
    NaNChecker(; schedule, fields)

Returns a `NaNChecker` that checks for a `NaN` anywhere in `fields` when `schedule` actuates.
`fields` should be a named tuple. The simulation is aborted if a `NaN` is found.
"""
NaNChecker(model=nothing; schedule, fields) = NaNChecker(schedule, fields)

function run_diagnostic!(nc::NaNChecker, model)
    for (name, field) in pairs(nc.fields)
        error_if_nan_in_field(field, name, model.clock)
    end
end

function error_if_nan_in_field(field, name, clock)
    if any(isnan, field.data.parent)
        error("time = $(clock.time), iteration = $(clock.iteration): NaN found in field $name. Aborting simulation.")
    end
end
