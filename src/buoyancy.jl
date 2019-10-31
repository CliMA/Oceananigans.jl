using .TurbulenceClosures: ∂x_faa, ▶x_faa, ∂y_afa, ▶y_afa, ∂z_aaf, ▶z_aaf

# https://en.wikipedia.org/wiki/Gravitational_acceleration#Gravity_model_for_Earth (30 Oct 2019)
const g_Earth = 9.80665

#=
Supported buoyancy types:

- Nothing 
- BuoyancyTracer
- SeawaterBuoyancy
=#

validate_buoyancy(::Nothing, tracers) = nothing

function validate_buoyancy(buoyancy, tracers) 
    req_tracers = required_tracers(buoyancy)

    all(tracer ∈ tracers for tracer in req_tracers) || 
        error("$(req_tracers) must be among the list of tracers to use $(typeof(buoyancy).name.wrapper)")

    return nothing
end

required_tracers(args...) = ()

#####
##### Functions for buoyancy = nothing
#####

@inline buoyancy_perturbation(i, j, k, grid::AbstractGrid{FT}, ::Nothing, C) where FT = zero(FT)

@inline ∂x_b(i, j, k, grid::AbstractGrid{FT}, ::Nothing, C) where FT = zero(FT)
@inline ∂y_b(i, j, k, grid::AbstractGrid{FT}, ::Nothing, C) where FT = zero(FT)
@inline ∂z_b(i, j, k, grid::AbstractGrid{FT}, ::Nothing, C) where FT = zero(FT)

@inline buoyancy_frequency_squared(args...) = ∂z_b(args...)

#####
##### Buoyancy as a tracer
#####

"""
    BuoyancyTracer <: AbstractBuoyancy{Nothing}

Type indicating that the tracer `b` represents buoyancy.
"""
struct BuoyancyTracer <: AbstractBuoyancy{Nothing} end

required_tracers(::BuoyancyTracer) = (:b,)

@inline buoyancy_perturbation(i, j, k, grid, ::BuoyancyTracer, C) = @inbounds C.b[i, j, k]

@inline ∂x_b(i, j, k, grid, ::BuoyancyTracer, C) = ∂x_faa(i, j, k, grid, C.b)
@inline ∂y_b(i, j, k, grid, ::BuoyancyTracer, C) = ∂y_afa(i, j, k, grid, C.b)
@inline ∂z_b(i, j, k, grid, ::BuoyancyTracer, C) = ∂z_aaf(i, j, k, grid, C.b)

"""
    SeawaterBuoyancy{G, EOS} <: AbstractBuoyancy{EOS}

Buoyancy model for temperature- and salt-stratified seawater.
"""
struct SeawaterBuoyancy{FT, EOS} <: AbstractBuoyancy{EOS}
    gravitational_acceleration :: FT
    equation_of_state :: EOS
end

"""
    SeawaterBuoyancy([FT=Float64;] gravitational_acceleration = g_Earth,
                                  equation_of_state = LinearEquationOfState(FT))

Returns parameters for a temperature- and salt-stratified seawater buoyancy model
with a `gravitational_acceleration` constant (typically called 'g'), and an
`equation_of_state` that related temperature and salinity (or conservative temperature
and absolute salinity) to density anomalies and buoyancy.
"""
function SeawaterBuoyancy(FT=Float64; 
                          gravitational_acceleration = g_Earth, 
                          equation_of_state = LinearEquationOfState(FT))
    return SeawaterBuoyancy{FT, typeof(equation_of_state)}(gravitational_acceleration, equation_of_state)
end

required_tracers(::SeawaterBuoyancy) = (:T, :S)

""" 
    ∂x_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the x-derivative of buoyancy for temperature and salt-stratified water,

```math
∂_x b = g ( α ∂_x Θ - β ∂_x Sᴬ ) ,
```

where `g` is gravitational acceleration, `α` is the thermal expansion
coefficient, `β` is the haline contraction coefficient, `Θ` is
conservative temperature, and `Sᴬ` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that `∂x_Θ`, `∂x_S`, `α`, and `β` are all evaluated at cell interfaces in `x`
and cell centers in `y` and `z`.
"""
@inline ∂x_b(i, j, k, grid, b::SeawaterBuoyancy, C) =
    b.gravitational_acceleration * (
           thermal_expansion_fcc(i, j, k, grid, b.equation_of_state, C) * ∂x_faa(i, j, k, grid, C.T)
        - haline_contraction_fcc(i, j, k, grid, b.equation_of_state, C) * ∂x_faa(i, j, k, grid, C.S) )

""" 
    ∂y_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the y-derivative of buoyancy for temperature and salt-stratified water,

```math
∂_y b = g ( α ∂_y Θ - β ∂_y Sᴬ ) ,
```

where `g` is gravitational acceleration, `α` is the thermal expansion
coefficient, `β` is the haline contraction coefficient, `Θ` is
conservative temperature, and `Sᴬ` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that `∂y_Θ`, `∂y_S`, `α`, and `β` are all evaluated at cell interfaces in `y` 
and cell centers in `x` and `z`.
"""
@inline ∂y_b(i, j, k, grid, b::SeawaterBuoyancy, C) =
    b.gravitational_acceleration * (
           thermal_expansion_cfc(i, j, k, grid, b.equation_of_state, C) * ∂y_afa(i, j, k, grid, C.T)
        - haline_contraction_cfc(i, j, k, grid, b.equation_of_state, C) * ∂y_afa(i, j, k, grid, C.S) )


""" 
    ∂z_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the vertical derivative of buoyancy for temperature and salt-stratified water,

```math
∂_z b = N^2 = g ( α ∂_z Θ - β ∂_z Sᴬ ) ,
```

where `g` is gravitational acceleration, `α` is the thermal expansion
coefficient, `β` is the haline contraction coefficient, `Θ` is
conservative temperature, and `Sᴬ` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that `∂z_Θ`, `∂z_Sᴬ`, `α`, and `β` are all evaluated at cell interfaces in `z`
and cell centers in `x` and `y`.
"""
@inline ∂z_b(i, j, k, grid, b::SeawaterBuoyancy, C) =
    b.gravitational_acceleration * (
           thermal_expansion_ccf(i, j, k, grid, b.equation_of_state, C) * ∂z_aaf(i, j, k, grid, C.T)
        - haline_contraction_ccf(i, j, k, grid, b.equation_of_state, C) * ∂z_aaf(i, j, k, grid, C.S) )

#####
##### Linear equation of state
#####

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
`β` [ppt⁻¹]. The buoyancy perturbation associated with a linear equation of state is

```math
    b = g (α T - β S)
```

Default constants are taken from Table 1.2 (page 33) of Vallis, "Atmospheric and Oceanic Fluid
Dynamics: Fundamentals and Large-Scale Circulation" (2ed, 2017).
"""
LinearEquationOfState(FT=Float64; α=1.67e-4, β=7.80e-4) = 
    LinearEquationOfState{FT}(α, β)

const LinearSeawaterBuoyancy = SeawaterBuoyancy{FT, <:LinearEquationOfState} where FT

@inline buoyancy_perturbation(i, j, k, grid, b::LinearSeawaterBuoyancy, C) =
    return @inbounds b.gravitational_acceleration * (   b.equation_of_state.α * C.T[i, j, k]
                                                      - b.equation_of_state.β * C.S[i, j, k] )

@inline  thermal_expansion(Θ, Sᴬ, D, eos::LinearEquationOfState) = eos.α
@inline haline_contraction(Θ, Sᴬ, D, eos::LinearEquationOfState) = eos.β

# Shortcuts
@inline  thermal_expansion_ccc(i, j, k, grid, eos::LinearEquationOfState, C) = eos.α
@inline  thermal_expansion_fcc(i, j, k, grid, eos::LinearEquationOfState, C) = eos.α
@inline  thermal_expansion_cfc(i, j, k, grid, eos::LinearEquationOfState, C) = eos.α
@inline  thermal_expansion_ccf(i, j, k, grid, eos::LinearEquationOfState, C) = eos.α

@inline haline_contraction_ccc(i, j, k, grid, eos::LinearEquationOfState, C) = eos.β
@inline haline_contraction_fcc(i, j, k, grid, eos::LinearEquationOfState, C) = eos.β
@inline haline_contraction_cfc(i, j, k, grid, eos::LinearEquationOfState, C) = eos.β
@inline haline_contraction_ccf(i, j, k, grid, eos::LinearEquationOfState, C) = eos.β

#####
##### Nonlinear equations of state
#####

""" Return the geopotential depth at `i, j, k` at cell centers. """
@inline D_aac(i, j, k, grid) = @inbounds -grid.zC[k]

""" Return the geopotential depth at `i, j, k` at cell z-interfaces. """
@inline D_aaf(i, j, k, grid) = @inbounds -grid.zF[k]

# Basic functionality
@inline ρ′(i, j, k, grid, eos, C) = @inbounds ρ′(C.T[i, j, k], C.S[i, j, k], D_aac(i, j, k, grid), eos)

@inline thermal_expansion_ccc(i, j, k, grid, eos, C) = @inbounds thermal_expansion(C.T[i, j, k], C.S[i, j, k], D_aac(i, j, k, grid), eos)
@inline thermal_expansion_fcc(i, j, k, grid, eos, C) = @inbounds thermal_expansion(▶x_faa(i, j, k, grid, C.T), ▶x_faa(i, j, k, grid, C.S), D_aac(i, j, k, grid), eos)
@inline thermal_expansion_cfc(i, j, k, grid, eos, C) = @inbounds thermal_expansion(▶y_afa(i, j, k, grid, C.T), ▶y_afa(i, j, k, grid, C.S), D_aac(i, j, k, grid), eos)
@inline thermal_expansion_ccf(i, j, k, grid, eos, C) = @inbounds thermal_expansion(▶z_aaf(i, j, k, grid, C.T), ▶z_aaf(i, j, k, grid, C.S), D_aaf(i, j, k, grid), eos)

@inline haline_contraction_ccc(i, j, k, grid, eos, C) = @inbounds haline_contraction(C.T[i, j, k], C.S[i, j, k], D_aac(i, j, k, grid), eos)
@inline haline_contraction_fcc(i, j, k, grid, eos, C) = @inbounds haline_contraction(▶x_faa(i, j, k, grid, C.T), ▶x_faa(i, j, k, grid, C.S), D_aac(i, j, k, grid), eos)
@inline haline_contraction_cfc(i, j, k, grid, eos, C) = @inbounds haline_contraction(▶y_afa(i, j, k, grid, C.T), ▶y_afa(i, j, k, grid, C.S), D_aac(i, j, k, grid), eos)
@inline haline_contraction_ccf(i, j, k, grid, eos, C) = @inbounds haline_contraction(▶z_aaf(i, j, k, grid, C.T), ▶z_aaf(i, j, k, grid, C.S), D_aaf(i, j, k, grid), eos)

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
