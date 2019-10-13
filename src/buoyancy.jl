using .TurbulenceClosures: ∂z_aaf, ▶z_aaf

const g_Earth = 9.80665

#=
Supported buoyancy types:

- Nothing
- SeawaterBuoyancy
=#

#####
##### Functions for buoyancy = nothing
#####

@inline buoyancy_perturbation(i, j, k, grid::AbstractGrid{T}, ::Nothing, C) where T = zero(T)
@inline buoyancy_frequency_squared(i, j, k, grid::AbstractGrid{T}, ::Nothing, C) where T = zero(T)

#####
##### Seawater buoyancy for buoyancy determined by temperature and salinity
#####

"""
    BuoyancyTracer <: AbstractBuoyancy{Nothing}

Type indicating that the tracer `T` represents buoyancy.
"""
struct BuoyancyTracer <: AbstractBuoyancy{Nothing} end

@inline buoyancy_perturbation(i, j, k, grid, ::BuoyancyTracer, C) = @inbounds C.T[i, j, k]
@inline buoyancy_frequency_squared(i, j, k, grid, ::BuoyancyTracer, C) = ∂z_aaf(i, j, k, grid, C.T)

"""
    SeawaterBuoyancy{G, EOS} <: AbstractBuoyancy{EOS}

Buoyancy model for temperature- and salt-stratified seawater.
"""
struct SeawaterBuoyancy{G, EOS} <: AbstractBuoyancy{EOS}
    gravitational_acceleration :: G
    equation_of_state :: EOS
end

"""
    SeawaterBuoyancy([T=Float64;] gravitational_acceleration = g_Earth,
                                  equation_of_state = LinearEquationOfState(T))

Returns parameters for a temperature- and salt-stratified seawater buoyancy model
with a `gravitational_acceleration` constant (typically called 'g'), and an
`equation_of_state` that related temperature and salinity (or conservative temperature
and absolute salinity) to density anomalies and buoyancy.
"""
function SeawaterBuoyancy(T=Float64;
                          gravitational_acceleration = g_Earth,
                          equation_of_state = LinearEquationOfState(T))
    return SeawaterBuoyancy{T, typeof(equation_of_state)}(gravitational_acceleration, equation_of_state)
end

"""
    buoyancy_frequency_squared(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the buoyancy frequency squared for temperature and salt-stratified water,

```math
N^2 = g \\left ( \\alpha \\partial_z T - \\beta \\partial_z S \\right ) \\, ,
```

where ``\$ g \$`` is gravitational acceleration, ``\$ \\alpha \$`` is the thermal expansion
coefficient, ``\$ \\beta \$`` is the haline contraction coefficient, ``\$ T \$`` is
temperature or conservative temperature, where applicable, and ``\$ S \$`` is the
salinity or absolute salinity, where applicable.
"""
@inline buoyancy_frequency_squared(i, j, k, grid, b::SeawaterBuoyancy, C) =
    b.gravitational_acceleration * (
           thermal_expansion(i, j, k, grid, b.equation_of_state, C) * ∂z_aaf(i, j, k, grid, C.T)
        - haline_contraction(i, j, k, grid, b.equation_of_state, C) * ∂z_aaf(i, j, k, grid, C.S) )

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
thermal expansion coefficient `α` [K⁻¹] and haline contraction coefficient
`β` [ppt⁻¹]. The buoyancy perturbation associated with a linear equation of state is

```math
    b = g (αT - βS)
```

Default constants are taken from Table 1.2 (page 33) of Vallis, "Atmospheric and Oceanic Fluid
Dynamics: Fundamentals and Large-Scale Circulation" (2ed, 2017).
"""
LinearEquationOfState(T=Float64; α=1.67e-4, β=7.80e-4) =
    LinearEquationOfState{T}(α, β)

const LinearSeawaterBuoyancy = SeawaterBuoyancy{FT, <:LinearEquationOfState} where FT

@inline buoyancy_perturbation(i, j, k, grid, b::LinearSeawaterBuoyancy, C) =
    return @inbounds b.gravitational_acceleration * (   b.equation_of_state.α * C.T[i, j, k]
                                                      - b.equation_of_state.β * C.S[i, j, k] )

@inline  thermal_expansion(i, j, k, grid, eos::LinearEquationOfState, C) = eos.α
@inline haline_contraction(i, j, k, grid, eos::LinearEquationOfState, C) = eos.β

#####
##### Nonlinear equations of state
#####

@inline buoyancy_perturbation(i, j, k, grid, b::AbstractBuoyancy{<:AbstractNonlinearEquationOfState}, C) =
    - b.gravitational_acceleration * ρ′(i, j, k, grid, b.equation_of_state, C) / b.ρ₀

#####
##### Roquet et al 2015 idealized nonlinear equations of state
#####

# Reference: Table 3 in Roquet et el, "Defining a Simplified yet 'Realistic' Equation of State for Seawater", (JPO, 2015)
optimized_roquet_coeffs = Dict(
                  :linear => (R₀₁₀ = -1.775e-1, R₁₀₀ = 7.718e-1, R₀₂₀ = 0,         R₀₁₁ = 0,          R₂₀₀ = 0,         R₁₀₁ = 0,         R₁₁₀ = 0),
               :cabbeling => (R₀₁₀ = -0.844e-1, R₁₀₀ = 7.718e-1, R₀₂₀ = -4.561e-3, R₀₁₁ = 0,          R₂₀₀ = 0,         R₁₀₁ = 0,         R₁₁₀ = 0),
:cabbeling_thermobaricity => (R₀₁₀ = -0.651e-1, R₁₀₀ = 7.718e-1, R₀₂₀ = -5.027e-3, R₀₁₁ = -2.5681e-5, R₂₀₀ = 0,         R₁₀₁ = 0,         R₁₁₀ = 0),
                :freezing => (R₀₁₀ = -0.491e-1, R₁₀₀ = 7.718e-1, R₀₂₀ = -5.027e-3, R₀₁₁ = -2.5681e-5, R₂₀₀ = 0,         R₁₀₁ = 0,         R₁₁₀ = 0),
            :second_order => (R₀₁₀ =  0.182e-1, R₁₀₀ = 8.078e-1, R₀₂₀ = -4.937e-3, R₀₁₁ = -2.4677e-5, R₂₀₀ = -1.115e-4, R₁₀₁ = -8.241e-6, R₁₁₀ = -2.446e-3)
)

"""
    RoquetIdealizedNonlinearEquationOfState{F, C, T} <: AbstractNonlinearEquationOfState

Parameters associated with the idealized nonlinear equation of state proposed by
Roquet et al., "Defining a Simplified yet 'Realistic' Equation of State for Seawater",
Journal of Physical Oceanography (2015).
"""
struct RoquetIdealizedNonlinearEquationOfState{F, C, T} <: AbstractNonlinearEquationOfState
                   ρ₀ :: T
    polynomial_coeffs :: C
end

type_convert_roquet_coeffs(T, coeffs) = NamedTuple{propertynames(coeffs)}(Tuple(T(R) for R in coeffs))

"""
    RoquetIdealizedNonlinearEquationOfState([T=Float64,] flavor, ρ₀=1024.6,
                                            polynomial_coeffs=optimized_roquet_coeffs[flavor])

Returns parameters for the idealized polynomial nonlinear equation of state with
reference density `ρ₀` and `polynomial_coeffs` proposed by Roquet et al., "Defining a
Simplified yet 'Realistic' Equation of State for Seawater", Journal of Physical
Oceanography (2015). The default reference density is `ρ₀ = 1024.6 kg m⁻³`, the average
surface density of seawater in the world ocean.

The `flavor` of the nonlinear equation of state is a symbol that selects a set of optimized
polynomial coefficients defined in Table 2 of Roquet et al., "Defining a Simplified yet
'Realistic' Equation of State for Seawater", Journal of Physical Oceanography (2015), and
further discussed in the text surrounding equations (12)--(15). The optimization minimizes
errors in estimated horizontal density gradient estiamted from climatological temperature
and salinity distributions between the 5 simplified forms chosen by Roquet et. al
and the full-fledged [TEOS-10](http://www.teos-10.org) equation of state.

The equations of state define the density anomaly `ρ′`, and have
the polynomial form

    `ρ′(T, S, D) = Σᵢⱼₐ Rᵢⱼₐ Tⁱ Sʲ Dᵃ`,

where `T` is conservative temperature, `S` is absolute salinity, and `D` is the
geopotential depth, currently just `D = -z`. The `Rᵢⱼₐ` are constant coefficients.

Flavors of idealized nonlinear equations of state
=================================================

    - `:linear`: a linear equation of state, `ρ′ = R₁₀₀ * T + R₀₁₀ * S`

    - `:cabbeling`: includes quadratic temperature term,
                    `ρ′ = R₁₀₀ * T + R₀₁₀ * S + R₀₂₀ * T^2`

    - `:cabbeling_thermobaricity`: includes 'thermobaricity' term,
                                   `ρ′ = R₁₀₀ * T + R₀₁₀ * S + R₀₂₀ * T^2 + R₀₁₁ * T * D`

    - `:freezing`: same as `:cabbeling_thermobaricity` with modified constants to increase
                   accuracy near freezing

    - `:second_order`: includes quadratic salinity, halibaricity, and thermohaline term,
                       `ρ′ = R₁₀₀ * T + R₀₁₀ * S + R₀₂₀ * T^2 + R₀₁₁ * T * D`
                             + R₂₀₀ * S^2 + R₁₀₁ * S * D + R₁₁₀ * S * T`

Example
=======

julia> using Oceananigans

julia> eos = Oceananigans.RoquetIdealizedNonlinearEquationOfState(:cabbeling);

julia> eos.polynomial_coeffs
(R₀₁₀ = -0.0844, R₁₀₀ = 0.7718, R₀₂₀ = -0.004561, R₀₁₁ = 0.0, R₂₀₀ = 0.0, R₁₀₁ = 0.0, R₁₁₀ = 0.0)

References
==========

    - Roquet et al., "Defining a Simplified yet 'Realistic' Equation of State for
      Seawater", Journal of Physical Oceanography (2015).

    - "Thermodynamic Equation of State for Seawater" (TEOS-10), http://www.teos-10.org
"""
function RoquetIdealizedNonlinearEquationOfState(T, flavor=:cabbeling_thermobaricity;
                                                 polynomial_coeffs=optimized_roquet_coeffs[flavor], ρ₀=1024.6)
    typed_coeffs = type_convert_roquet_coeffs(T, polynomial_coeffs)
    return RoquetIdealizedNonlinearEquationOfState{flavor, typeof(typed_coeffs), T}(ρ₀, typed_coeffs)
end

RoquetIdealizedNonlinearEquationOfState(flavor::Symbol=:cabbeling_thermobaricity; kwargs...) =
    RoquetIdealizedNonlinearEquationOfState(Float64, flavor; kwargs...)

""" Return the geopotential depth at `i, j, k` at cell centers. """
@inline D_aac(i, j, k, grid) = @inbounds -grid.zC[k]
const D = D_aac

""" Return the geopotential depth at `i, j, k` at cell z-interfaces. """
@inline D_aaf(i, j, k, grid) = @inbounds -grid.zF[k]

@inline ρ′(i, j, k, grid, eos::RoquetIdealizedNonlinearEquationOfState, C) =
    @inbounds (   eos.polynomial_coeffs.R₁₀₀ * C.S[i, j, k]
                + eos.polynomial_coeffs.R₀₁₀ * C.T[i, j, k]
                + eos.polynomial_coeffs.R₀₂₀ * C.T[i, j, k]^2
                + eos.polynomial_coeffs.R₀₁₁ * C.T[i, j, k] * D(i, j, k, grid)
                + eos.polynomial_coeffs.R₂₀₀ * C.S[i, j, k]^2
                + eos.polynomial_coeffs.R₁₀₁ * C.S[i, j, k] * D(i, j, k, grid)
                + eos.polynomial_coeffs.R₁₁₀ * C.S[i, j, k] * C.T[i, j, k] )

@inline thermal_expansion(i, j, k, grid, eos::RoquetIdealizedNonlinearEquationOfState, C) =
    @inbounds (   eos.polynomial_coeffs.R₀₁₀
                + 2 * eos.polynomial_coeffs.R₀₂₀ * ▶z_aaf(i, j, k, grid, C.T)
                + eos.polynomial_coeffs.R₀₁₁ * D_aaf(i, j, k, grid)
                + eos.polynomial_coeffs.R₁₁₀ * ▶z_aaf(i, j, k, grid, C.T) )

@inline haline_contraction(i, j, k, grid, eos::RoquetIdealizedNonlinearEquationOfState, C) =
    @inbounds (   eos.polynomial_coeffs.R₁₀₀
                + 2 * eos.polynomial_coeffs.R₂₀₀ * ▶z_aaf(i, j, k, grid, C.S)
                + eos.polynomial_coeffs.R₁₀₁ * D_aaf(i, j, k, grid)
                + eos.polynomial_coeffs.R₁₁₀ * ▶z_aaf(i, j, k, grid, C.S) )
