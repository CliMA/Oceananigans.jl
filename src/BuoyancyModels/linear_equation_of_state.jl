"""
    LinearEquationOfState{FT} <: AbstractEquationOfState

Linear equation of state for seawater.
"""
struct LinearEquationOfState{FT} <: AbstractEquationOfState
    thermal_expansion :: FT
    haline_contraction :: FT
end

Base.summary(eos::LinearEquationOfState) =
    string("LinearEquationOfState(thermal_expansion=", prettysummary(eos.thermal_expansion),
                               ", haline_contraction=", prettysummary(eos.haline_contraction), ")")

Base.show(io, eos::LinearEquationOfState) = print(io, summary(eos))

"""
    LinearEquationOfState([FT=Float64;] thermal_expansion=1.67e-4, haline_contraction=7.80e-4)

Return `LinearEquationOfState` for `SeawaterBuoyancy` with
`thermal_expansion` coefficient and `haline_contraction` coefficient.
The buoyancy perturbation ``b`` for `LinearEquationOfState` is

```math
    b = g (α T - β S),
```

where ``g`` is gravitational acceleration, ``α`` is `thermal_expansion`, ``β`` is
`haline_contraction`, ``T`` is temperature, and ``S`` is practical salinity units.

Default constants in units inverse Kelvin and practical salinity units
for `thermal_expansion` and `haline_contraction`, respectively,
are taken from Table 1.2 (page 33) of Vallis, "Atmospheric and Oceanic Fluid
Dynamics: Fundamentals and Large-Scale Circulation" (2nd ed, 2017).
"""
LinearEquationOfState(FT=Float64; thermal_expansion=1.67e-4, haline_contraction=7.80e-4) =
    LinearEquationOfState{FT}(thermal_expansion, haline_contraction)

#####
##### Thermal expansion and haline contraction coefficients
#####

@inline  thermal_expansion(Θ, sᴬ, D, eos::LinearEquationOfState) = eos.thermal_expansion
@inline haline_contraction(Θ, sᴬ, D, eos::LinearEquationOfState) = eos.haline_contraction

# Shortcuts
@inline  thermal_expansionᶜᶜᶜ(i, j, k, grid, eos::LinearEquationOfState, C) = eos.thermal_expansion
@inline  thermal_expansionᶠᶜᶜ(i, j, k, grid, eos::LinearEquationOfState, C) = eos.thermal_expansion
@inline  thermal_expansionᶜᶠᶜ(i, j, k, grid, eos::LinearEquationOfState, C) = eos.thermal_expansion
@inline  thermal_expansionᶜᶜᶠ(i, j, k, grid, eos::LinearEquationOfState, C) = eos.thermal_expansion

@inline haline_contractionᶜᶜᶜ(i, j, k, grid, eos::LinearEquationOfState, C) = eos.haline_contraction
@inline haline_contractionᶠᶜᶜ(i, j, k, grid, eos::LinearEquationOfState, C) = eos.haline_contraction
@inline haline_contractionᶜᶠᶜ(i, j, k, grid, eos::LinearEquationOfState, C) = eos.haline_contraction
@inline haline_contractionᶜᶜᶠ(i, j, k, grid, eos::LinearEquationOfState, C) = eos.haline_contraction

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
    @inbounds b.gravitational_acceleration * (b.equation_of_state.thermal_expansion * C.T[i, j, k] -
                                              b.equation_of_state.haline_contraction * C.S[i, j, k])

@inline buoyancy_perturbation(i, j, k, grid, b::LinearTemperatureSeawaterBuoyancy, C) =
    @inbounds b.gravitational_acceleration * b.equation_of_state.thermal_expansion * C.T[i, j, k]

@inline buoyancy_perturbation(i, j, k, grid, b::LinearSalinitySeawaterBuoyancy, C) =
    @inbounds - b.gravitational_acceleration * b.equation_of_state.haline_contraction * C.S[i, j, k]

