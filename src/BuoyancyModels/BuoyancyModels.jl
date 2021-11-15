module BuoyancyModels

export
    Buoyancy, BuoyancyTracer, SeawaterBuoyancy, buoyancy_perturbation,
    LinearEquationOfState, RoquetIdealizedNonlinearEquationOfState, TEOS10,
    ∂x_b, ∂y_b, ∂z_b, buoyancy_perturbation, x_dot_g_b, y_dot_g_b, z_dot_g_b,
    top_buoyancy_flux,
    buoyancy_frequency_squared,
    BuoyancyField

using Printf
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.BoundaryConditions: getbc

import SeawaterPolynomials: ρ′, thermal_expansion, haline_contraction

# Physical constants
# https://en.wikipedia.org/wiki/Gravitational_acceleration#Gravity_model_for_Earth (30 Oct 2019)
const g_Earth = 9.80665

"""
    AbstractBuoyancyModel{EOS}

Abstract supertype for buoyancy models.
"""
abstract type AbstractBuoyancyModel{EOS} end

"""
    AbstractEquationOfState

Abstract supertype for equations of state.
"""
abstract type AbstractEquationOfState end

function validate_buoyancy(buoyancy, tracers)
    req_tracers = required_tracers(buoyancy)

    all(tracer ∈ tracers for tracer in req_tracers) ||
        error("$(req_tracers) must be among the list of tracers to use $(typeof(buoyancy).name.wrapper)")

    return nothing
end

include("buoyancy.jl")
include("no_buoyancy.jl")
include("buoyancy_tracer.jl")
include("seawater_buoyancy.jl")
include("linear_equation_of_state.jl")
include("nonlinear_equation_of_state.jl")
include("g_dot_b.jl")
include("buoyancy_field.jl")

Base.show(io::IO, b::SeawaterBuoyancy{FT}) where FT =
    print(io, "SeawaterBuoyancy{$FT}: g = $(b.gravitational_acceleration)", '\n',
                "└── equation of state: $(b.equation_of_state)")

Base.show(io::IO, eos::LinearEquationOfState{FT}) where FT =
    print(io, "LinearEquationOfState{$FT}: ", @sprintf("α = %.2e, β = %.2e", eos.α, eos.β))

end
