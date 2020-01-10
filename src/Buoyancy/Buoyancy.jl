module Buoyancy

export
    BuoyancyTracer, SeawaterBuoyancy, buoyancy_perturbation,
    LinearEquationOfState, RoquetIdealizedNonlinearEquationOfState

# Physical constants
# https://en.wikipedia.org/wiki/Gravitational_acceleration#Gravity_model_for_Earth (30 Oct 2019)
const g_Earth = 9.80665

"""
    AbstractBuoyancy{EOS}

Abstract supertype for buoyancy models.
"""
abstract type AbstractBuoyancy{EOS} end

"""
    AbstractEquationOfState

Abstract supertype for equations of state.
"""
abstract type AbstractEquationOfState end

"""
    AbstractNonlinearEquationOfState

Abstract supertype for nonlinar equations of state.
"""
abstract type AbstractNonlinearEquationOfState <: AbstractEquationOfState end

include("buoyancy_utils.jl")
include("no_buoyancy.jl")
include("buoyancy_tracer.jl")
include("seawater_buoyancy.jl")
include("linear_equation_of_state.jl")
include("nonlinear_equation_of_state.jl")
include("roquet_idealized_nonlinear_eos.jl")
include("show_buoyancy.jl")

end
