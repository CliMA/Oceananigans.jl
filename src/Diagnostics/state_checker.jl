using Printf
using Statistics

struct StateChecker{T, F} <: AbstractDiagnostic
    schedule :: T
      fields :: F
end

"""
    StateChecker(; schedule, fields)

Returns a `StateChecker` that logs field information (minimum, maximum, mean)
for each field in a named tuple of `fields` when `schedule` actuates.
"""
StateChecker(model; schedule, fields=fields(model)) = StateChecker(schedule, fields)

function run_diagnostic!(sc::StateChecker, model)
    pad = keys(sc.fields) .|> string .|> length |> maximum

    @info "State check @ $(summary(model.clock))"

    for (name, field) in pairs(sc.fields)
        state_check(field, name, pad)
    end

    return nothing
end

function state_check(field, name, pad)
    min_val = minimum(field)
    max_val = maximum(field)
    mean_val = mean(field)

    name = lpad(name, pad)

    @info @sprintf("%s | min = %+.15e | max = %+.15e | mean = %+.15e", name, min_val, max_val, mean_val)
    return nothing
end

(sc::StateChecker)(model) = run_diagnostic!(sc, model)

Base.show(io::IO, sc::StateChecker) =
    print(io, "StateChecker checking $(length(sc.fields)) fields: $(keys(sc.fields))")
