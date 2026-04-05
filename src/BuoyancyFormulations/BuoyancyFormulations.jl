module BuoyancyFormulations

export
    BuoyancyForce, BuoyancyTracer, SeawaterBuoyancy, LinearEquationOfState,
    ∂x_b, ∂y_b, ∂z_b, buoyancy_perturbationᶜᶜᶜ, x_dot_g_bᶠᶜᶜ, y_dot_g_bᶜᶠᶜ, z_dot_g_bᶜᶜᶠ,
    top_buoyancy_flux

using Oceananigans: Oceananigans
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, ℑxyᶠᶠᵃ, ℑxyzᶠᶠᶠ, ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ, ∂zᶠᶠᶠ
using Oceananigans.BoundaryConditions: getbc

using SeawaterPolynomials: SeawaterPolynomials, ρ′, thermal_expansion, haline_contraction, with_float_type

"""
    AbstractBuoyancyFormulation{EOS}

Abstract supertype for buoyancy models.
"""
abstract type AbstractBuoyancyFormulation{EOS} end

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

include("buoyancy_force.jl")
include("no_buoyancy.jl")
include("buoyancy_tracer.jl")
include("seawater_buoyancy.jl")
include("linear_equation_of_state.jl")
include("nonlinear_equation_of_state.jl")
include("g_dot_b.jl")

end # module
