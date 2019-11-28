abstract type AbstractPrognosticTemperature end

struct Temperature <: AbstractPrognosticTemperature end
struct ModifiedPotentialTemperature <: AbstractPrognosticTemperature end
struct Entropy <: AbstractPrognosticTemperature end

function validate_prognostic_temperature(::Temperature, tracers)
    if :T ∉ tracers
        throw(ArgumentError("Must specify a T tracer to use Temperature as a prognostic variable."))
    end
    return nothing
end

function validate_prognostic_temperature(::ModifiedPotentialTemperature, tracers)
    if :Θᵐ ∉ tracers
        throw(ArgumentError("Must specify a Θᵐ tracer to use ModifiedPotentialTemperature as a prognostic variable."))
    end
    return nothing
end

function validate_prognostic_temperature(::Entropy, tracers)
    if :S ∉ tracers
        throw(ArgumentError("Must specify S as a tracer to use Entropy as a prognostic variable."))
    end
    return nothing
end

