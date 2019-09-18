using .TurbulenceClosures: ∂z_aaf

abstract type AbstractBuoyancy end

const Earth_gravitational_acceleration = 9.80665

#=
Supported buoyancy types:

- Nothing 
- SeawaterBuoyancy
=#

#####
##### Functions for buoyancy = nothing
#####

@inline buoyancy_perturbation(i, j, k, grid::AbstractGrid{T}, ::Nothing, C) where T = zero(T)
@inline total_buoyancy(i, j, k, grid::AbstractGrid{T}, ::Nothing, C) where T = zero(T)
@inline N²(i, j, k, grid::AbstractGrid{T}, ::Nothing, C) where T = zero(T)

#####
##### Seawater buoyancy for buoyancy determined by temperature and salinity
#####

struct SeawaterBuoyancy{G, EOS} <: AbstractBuoyancy
    gravitational_acceleration :: G
    equation_of_state :: EOS
end

function SeawaterBuoyancy(T=Float64; 
                          gravitational_acceleration = Earth_gravitational_acceleration, 
                          equation_of_state = LinearEquationOfState(T))
    return SeawaterBuoyancy{T, typeof(equation_of_state)}(gravitational_acceleration, equation_of_state)
end

#####
##### Generic functions for equations of state
#####

@inline N²(i, j, k, grid, buoyancy::SeawaterBuoyancy, C) = 
    buoyancy.gravitational_acceleration * (
             thermal_expansion(i, j, k, grid, buoyancy.equation_of_state, C) * ∂z_aaf(i, j, k, grid, C.T)
          - haline_contraction(i, j, k, grid, buoyancy.equation_of_state, C) * ∂z_aaf(i, j, k, grid, C.S))

#####
##### Linear equation of state
#####

"""
    LinearEquationOfState{T} <: AbstractEquationOfState

Linear equation of state for seawater. 
"""
struct LinearEquationOfState{T} <: AbstractEquationOfState
    α :: T
    β :: T
end

"""
    LinearEquationOfState([T=Float64;] α=1.67e-4, β=7.80e-4)

Returns parameters for a linear equation of state for seawater with
thermal expansion coefficient `α` and haline contraction coefficient `β`.
The dynamic component of the buoyancy perturbation associated 
with a linear equation of state is

    `b′ = g * α * T - g * β * S`

Default constants are taken from Table 1.2 (page 33) of Vallis, "Atmospheric and Oceanic Fluid 
Dynamics: Fundamentals and Large-Scale Circulation" (2ed, 2017).
"""
LinearEquationOfState(T=Float64; α=1.67e-4, β=7.80e-4) = 
    LinearEquationOfState{T}(α, β)

const LinearSeawaterBuoyancy = SeawaterBuoyancy{FT, <:LinearEquationOfState} where FT

@inline function buoyancy_perturbation(i, j, k, grid, buoyancy::LinearSeawaterBuoyancy, C)
    return @inbounds buoyancy.gravitational_acceleration * (   
                          buoyancy.equation_of_state.α * C.T[i, j, k]
                        - buoyancy.equation_of_state.β * C.S[i, j, k] )
end

@inline total_buoyancy(i, j, k, grid, buoyancy::LinearSeawaterBuoyancy, C) =
    buoyancy_perturbation(i, j, k, grid, buoyancy, C)

@inline  thermal_expansion(i, j, k, grid, eos::LinearEquationOfState, C) = eos.α
@inline haline_contraction(i, j, k, grid, eos::LinearEquationOfState, C) = eos.β
