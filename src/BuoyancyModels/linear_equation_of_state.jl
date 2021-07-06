"""
    LinearEquationOfState{FT} <: AbstractEquationOfState

Linear equation of state for seawater.
"""
struct LinearEquationOfState{FT} <: AbstractEquationOfState
    α :: FT
    β :: FT
end

"""
    LinearEquationOfState([FT=Float64;] α=1.67e-4, β=7.80e-4)

Returns parameters for a linear equation of state for seawater with
thermal expansion coefficient `α` [K⁻¹] and haline contraction coefficient
`β` [psu⁻¹]. The buoyancy perturbation associated with a linear equation of state is

```math
    b = g (α T - β S)
```

Default constants are taken from Table 1.2 (page 33) of Vallis, "Atmospheric and Oceanic Fluid
Dynamics: Fundamentals and Large-Scale Circulation" (2ed, 2017).
"""
LinearEquationOfState(FT=Float64; α=1.67e-4, β=7.80e-4) = LinearEquationOfState{FT}(α, β)

#####
##### Thermal expansion and haline contraction coefficients
#####

@inline  thermal_expansion(Θ, sᴬ, D, eos::LinearEquationOfState) = eos.α
@inline haline_contraction(Θ, sᴬ, D, eos::LinearEquationOfState) = eos.β

# Shortcuts
@inline  thermal_expansionᶜᶜᶜ(i, j, k, grid, eos::LinearEquationOfState, C) = eos.α
@inline  thermal_expansionᶠᶜᶜ(i, j, k, grid, eos::LinearEquationOfState, C) = eos.α
@inline  thermal_expansionᶜᶠᶜ(i, j, k, grid, eos::LinearEquationOfState, C) = eos.α
@inline  thermal_expansionᶜᶜᶠ(i, j, k, grid, eos::LinearEquationOfState, C) = eos.α

@inline haline_contractionᶜᶜᶜ(i, j, k, grid, eos::LinearEquationOfState, C) = eos.β
@inline haline_contractionᶠᶜᶜ(i, j, k, grid, eos::LinearEquationOfState, C) = eos.β
@inline haline_contractionᶜᶠᶜ(i, j, k, grid, eos::LinearEquationOfState, C) = eos.β
@inline haline_contractionᶜᶜᶠ(i, j, k, grid, eos::LinearEquationOfState, C) = eos.β

#####
##### Convinient aliases to dispatch on
#####

const LinearSeawaterBuoyancy = SeawaterBuoyancy{FT, <:LinearEquationOfState} where FT
const LinearTemperatureSeawaterBuoyancy = SeawaterBuoyancy{FT, <:LinearEquationOfState, <:Nothing, <:Number} where FT
const LinearSalinitySeawaterBuoyancy = SeawaterBuoyancy{FT, <:LinearEquationOfState, <:Number, <:Nothing} where FT

#####
##### BuoyancyModels perturbation
#####

@inline buoyancy_perturbation(i, j, k, grid, b::LinearSeawaterBuoyancy, C) =
    @inbounds b.gravitational_acceleration * (b.equation_of_state.α * C.T[i, j, k] - b.equation_of_state.β * C.S[i, j, k])

@inline buoyancy_perturbation(i, j, k, grid, b::LinearTemperatureSeawaterBuoyancy, C) =
    @inbounds b.gravitational_acceleration * b.equation_of_state.α * C.T[i, j, k]

@inline buoyancy_perturbation(i, j, k, grid, b::LinearSalinitySeawaterBuoyancy, C) =
    @inbounds - b.gravitational_acceleration * b.equation_of_state.β * C.S[i, j, k]
