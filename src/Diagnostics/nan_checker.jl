struct NaNChecker{T, F} <: AbstractDiagnostic
    schedule :: T
      fields :: F
end

"""
    NaNChecker(; schedule, fields)

Returns a `NaNChecker` that checks for a `NaN` in the first grid points of `fields`
when `schedule` actuates. `fields` should be a named tuple.

The simulation is aborted if a `NaN` is found.
"""
NaNChecker(model=nothing; schedule, fields) = NaNChecker(schedule, fields)

function run_diagnostic!(nc::NaNChecker, model)
    for (name, field) in pairs(nc.fields)
        CUDA.@allowscalar begin
            if isnan(field.data[1, 1, 1])
                t, i = model.clock.time, model.clock.iteration
                error("time = $t, iteration = $i: NaN found in $name. Aborting simulation.")
            end
        end
    end
end
