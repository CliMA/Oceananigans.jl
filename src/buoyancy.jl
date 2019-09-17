abstract type AbstractBuoyancy end

const g_Earth = 9.80665

struct SeawaterBuoyancy{T, EOS} <: AbstractBuoyancy
    g :: T
    equation_of_state :: EOS
end

function SeawaterBuoyancy(T=Float64; g=g_Earth, equation_of_state=LinearEquationOfState(T))
    return SeawaterBuoyancy{T, typeof(equation_of_state)}(g, equation_of_state)
end

@inline buoyancy_perturbation(i, j, k, grid::AbstractGrid{T}, ::Nothing, C) where T = zero(T)
@inline total_buoyancy(i, j, k, grid::AbstractGrid{T}, ::Nothing, C) where T = zero(T)

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

Returns parameters for a linear equation of state for seawater, where
`α` is the thermal expansion coefficient and `β` is the haline contraction
coefficient. The dynamic component of the buoyancy perturbation associated 
with a linear equation of state is

    `b′ = α * T - β * S`

Default constants are taken from Eq. (1.57) of Vallis, "Atmospheric and Oceanic Fluid 
Dynamics: Fundamentals and Large-Scale Circulation" (2ed, 2017).
"""
LinearEquationOfState(T=Float64; α=1.67e-4, β=7.80e-4) = LinearEquationOfState{T}(α, β)

const LinearSeawaterBuoyancy = SeawaterBuoyancy{FT, <:LinearEquationOfState} where FT

@inline function buoyancy_perturbation(i, j, k, grid, buoyancy::LinearSeawaterBuoyancy, C)
    return @inbounds buoyancy.g * (   buoyancy.equation_of_state.α * C.T[i, j, k]
                                    - buoyancy.equation_of_state.β * C.S[i, j, k] )
end

@inline total_buoyancy(i, j, k, grid, buoyancy::LinearSeawaterBuoyancy, C) =
    buoyancy_perturbation(i, j, k, grid, buoyancy, C)
