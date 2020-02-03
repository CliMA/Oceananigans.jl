abstract type AbstractThermodynamicVariable end

struct ModifiedPotentialTemperature <: AbstractThermodynamicVariable end
struct Entropy <: AbstractThermodynamicVariable end

missing_tracer_error(name, tvar) =
    "Must specify a $name tracer to use $(typeof(tvar)) as a thermodynamic variable."

required_tracer(::ModifiedPotentialTemperature) = :Θᵐ
required_tracer(::Entropy) = :S

function validate_thermodynamic_variable(thermodynamic_variable, tracers)
    c = required_tracer(thermodynamic_variable)
    if c ∉ tracers
        throw(ArgumentError(missing_tracer_error(c, thermodynamic_variable)))
    else
        return true
    end
end
