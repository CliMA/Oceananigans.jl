# Reference: Table 3 in Roquet et. al., "Defining a Simplified yet 'Realistic'
# Equation of State for Seawater", Journal of Physical Oceanography (2015).
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
struct RoquetIdealizedNonlinearEquationOfState{F, C, FT} <: AbstractNonlinearEquationOfState
                   ρ₀ :: FT
    polynomial_coeffs :: C
end

type_convert_roquet_coeffs(FT, coeffs) = NamedTuple{propertynames(coeffs)}(Tuple(FT(R) for R in coeffs))

"""
    RoquetIdealizedNonlinearEquationOfState([FT=Float64,] flavor, ρ₀=1024.6,
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
function RoquetIdealizedNonlinearEquationOfState(FT, flavor=:cabbeling_thermobaricity;
                                                 polynomial_coeffs=optimized_roquet_coeffs[flavor], ρ₀=1024.6)
    typed_coeffs = type_convert_roquet_coeffs(FT, polynomial_coeffs)
    return RoquetIdealizedNonlinearEquationOfState{flavor, typeof(typed_coeffs), FT}(ρ₀, typed_coeffs)
end

RoquetIdealizedNonlinearEquationOfState(flavor::Symbol=:cabbeling_thermobaricity; kwargs...) =
    RoquetIdealizedNonlinearEquationOfState(Float64, flavor; kwargs...)

@inline ρ′(Θ, Sᴬ, D, eos::RoquetIdealizedNonlinearEquationOfState) =
    @inbounds (   eos.polynomial_coeffs.R₁₀₀ * Sᴬ
                + eos.polynomial_coeffs.R₀₁₀ * Θ
                + eos.polynomial_coeffs.R₀₂₀ * Θ^2
                + eos.polynomial_coeffs.R₀₁₁ * Θ * D
                + eos.polynomial_coeffs.R₂₀₀ * Sᴬ^2
                + eos.polynomial_coeffs.R₁₀₁ * Sᴬ * D
                + eos.polynomial_coeffs.R₁₁₀ * Sᴬ * Θ )

@inline thermal_expansion(Θ, Sᴬ, D, eos::RoquetIdealizedNonlinearEquationOfState) =
    @inbounds (       eos.polynomial_coeffs.R₀₁₀
                + 2 * eos.polynomial_coeffs.R₀₂₀ * Θ
                +     eos.polynomial_coeffs.R₀₁₁ * D
                +     eos.polynomial_coeffs.R₁₁₀ * Sᴬ )

@inline haline_contraction(Θ, Sᴬ, D, eos::RoquetIdealizedNonlinearEquationOfState) =
    @inbounds (       eos.polynomial_coeffs.R₁₀₀
                + 2 * eos.polynomial_coeffs.R₂₀₀ * Sᴬ
                    + eos.polynomial_coeffs.R₁₀₁ * D
                    + eos.polynomial_coeffs.R₁₁₀ * Θ )
