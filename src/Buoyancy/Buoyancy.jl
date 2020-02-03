module Buoyancy

export
    BuoyancyTracer, SeawaterBuoyancy, buoyancy_perturbation,
    LinearEquationOfState, RoquetIdealizedNonlinearEquationOfState,
    ∂x_b, ∂y_b, ∂z_b, buoyancy_perturbation, buoyancy_frequency_squared

using Printf
using Oceananigans.Grids
using Oceananigans.Operators

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

function validate_buoyancy(buoyancy, tracers)
    req_tracers = required_tracers(buoyancy)

    all(tracer ∈ tracers for tracer in req_tracers) ||
        error("$(req_tracers) must be among the list of tracers to use $(typeof(buoyancy).name.wrapper)")

    return nothing
end

validate_buoyancy(::Nothing, tracers) = nothing

required_tracers(::Nothing) = ()

@inline buoyancy_perturbation(i, j, k, grid::AbstractGrid{FT}, ::Nothing, C) where FT = zero(FT)

@inline ∂x_b(i, j, k, grid::AbstractGrid{FT}, ::Nothing, C) where FT = zero(FT)
@inline ∂y_b(i, j, k, grid::AbstractGrid{FT}, ::Nothing, C) where FT = zero(FT)
@inline ∂z_b(i, j, k, grid::AbstractGrid{FT}, ::Nothing, C) where FT = zero(FT)

"""
    BuoyancyTracer <: AbstractBuoyancy{Nothing}

Type indicating that the tracer `b` represents buoyancy.
"""
struct BuoyancyTracer <: AbstractBuoyancy{Nothing} end

required_tracers(::BuoyancyTracer) = (:b,)

@inline buoyancy_perturbation(i, j, k, grid, ::BuoyancyTracer, C) = @inbounds C.b[i, j, k]

@inline ∂x_b(i, j, k, grid, ::BuoyancyTracer, C) = ∂xᶠᵃᵃ(i, j, k, grid, C.b)
@inline ∂y_b(i, j, k, grid, ::BuoyancyTracer, C) = ∂yᵃᶠᵃ(i, j, k, grid, C.b)
@inline ∂z_b(i, j, k, grid, ::BuoyancyTracer, C) = ∂zᵃᵃᶠ(i, j, k, grid, C.b)

include("seawater_buoyancy.jl")
include("linear_equation_of_state.jl")
include("nonlinear_equation_of_state.jl")
include("roquet_idealized_nonlinear_eos.jl")

Base.show(io::IO, b::SeawaterBuoyancy{FT}) where FT =
    println(io, "SeawaterBuoyancy{$FT}: g = $(b.gravitational_acceleration)", '\n',
                "└── equation of state: $(b.equation_of_state)")

Base.show(io::IO, eos::LinearEquationOfState{FT}) where FT =
    println(io, "LinearEquationOfState{$FT}: ", @sprintf("α = %.2e, β = %.2e", eos.α, eos.β))

end
