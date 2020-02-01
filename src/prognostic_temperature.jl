abstract type AbstractPrognosticTemperature end

struct Temperature <: AbstractPrognosticTemperature end
struct ModifiedPotentialTemperature <: AbstractPrognosticTemperature end
struct Entropy <: AbstractPrognosticTemperature end

missing_tracer_error(name, pt) =
    "Must specify a $name tracer to use $(typeof(pt)) as a prognostic temperature variable."

required_tracer(::Temperature) = :T
required_tracer(::ModifiedPotentialTemperature) = :Θᵐ
required_tracer(::Entropy) = :S

function validate_prognostic_temperature(prognostic_temperature, tracers)
    c = required_tracer(prognostic_temperature)
    if c ∉ tracers
        throw(ArgumentError(missing_tracer_error(c, prognostic_temperature)))
    else
        return true
    end
end
