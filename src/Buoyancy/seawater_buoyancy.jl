"""
    SeawaterBuoyancy{FT, EOS, T, S} <: AbstractBuoyancy{EOS}

Buoyancy model for seawater. `T` and `S` are either `nothing` if both
temperature and salinity are active, or of type `FT` if temperature
or salinity are constant, respectively.
"""
struct SeawaterBuoyancy{FT, EOS, T, S} <: AbstractBuoyancy{EOS}
    gravitational_acceleration :: FT
             equation_of_state :: EOS
          constant_temperature :: T
             constant_salinity :: S
end

required_tracers(::SeawaterBuoyancy) = (:T, :S)
required_tracers(::SeawaterBuoyancy{FT, EOS, <:Nothing, <:Number}) where {FT, EOS} = (:T,) # active temperature only
required_tracers(::SeawaterBuoyancy{FT, EOS, <:Number, <:Nothing}) where {FT, EOS} = (:S,) # active salinity only

"""
    SeawaterBuoyancy([FT=Float64;] gravitational_acceleration = g_Earth,
                                  equation_of_state = LinearEquationOfState(FT), 
                                  constant_temperature = false, constant_salinity = false)

Returns parameters for a temperature- and salt-stratified seawater buoyancy model
with a `gravitational_acceleration` constant (typically called 'g'), and an
`equation_of_state` that related temperature and salinity (or conservative temperature
and absolute salinity) to density anomalies and buoyancy. If either `temperature` or `salinity`
are specified, buoyancy is calculated
"""
function SeawaterBuoyancy(                        FT = Float64;
                          gravitational_acceleration = g_Earth,
                                   equation_of_state = LinearEquationOfState(FT),
                                constant_temperature = nothing,
                                   constant_salinity = nothing)

    # Input validation: convert constant_temperature or constant_salinity = true to zero(FT). 
    # This method of specifying constant temperature or salinity in a SeawaterBuoyancy model
    # should only be used with a LinearEquationOfState where the constant value of either temperature
    # or sailnity is irrelevant.
    constant_temperature = constant_temperature === true ? zero(FT) : constant_temperature
    constant_salinity = constant_salinity === true ? zero(FT) : constant_salinity

    return SeawaterBuoyancy{FT, typeof(equation_of_state), typeof(constant_temperature), typeof(constant_salinity)}(
                            gravitational_acceleration, equation_of_state, constant_temperature, constant_salinity)
end

const TemperatureSeawaterBuoyancy = SeawaterBuoyancy{FT, EOS, <:Nothing, <:Number} where {FT, EOS}
const SalinitySeawaterBuoyancy = SeawaterBuoyancy{FT, EOS, <:Number, <:Nothing} where {FT, EOS}

@inline get_temperature_and_salinity(::SeawaterBuoyancy, C) = C.T, C.S
@inline get_temperature_and_salinity(b::TemperatureSeawaterBuoyancy, C) = C.T, b.constant_salinity
@inline get_temperature_and_salinity(b::SalinitySeawaterBuoyancy, C) = b.constant_temperature, C.S

"""
    ∂x_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the x-derivative of buoyancy for temperature and salt-stratified water,

```math
∂_x b = g ( α ∂_x Θ - β ∂_x sᴬ ) ,
```

where `g` is gravitational acceleration, `α` is the thermal expansion
coefficient, `β` is the haline contraction coefficient, `Θ` is
conservative temperature, and `sᴬ` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that `∂x_Θ`, `∂x_sᴬ`, `α`, and `β` are all evaluated at cell interfaces in `x`
and cell centers in `y` and `z`.
"""
@inline function ∂x_b(i, j, k, grid, b::SeawaterBuoyancy, C)
    Θ, sᴬ = get_temperature_and_salinity(b, C)
    return b.gravitational_acceleration * (
           thermal_expansionᶠᶜᶜ(i, j, k, grid, b.equation_of_state, Θ, sᴬ) * ∂xᶠᵃᵃ(i, j, k, grid, Θ)
        - haline_contractionᶠᶜᶜ(i, j, k, grid, b.equation_of_state, Θ, sᴬ) * ∂xᶠᵃᵃ(i, j, k, grid, sᴬ) )
end

"""
    ∂y_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the y-derivative of buoyancy for temperature and salt-stratified water,

```math
∂_y b = g ( α ∂_y Θ - β ∂_y sᴬ ) ,
```

where `g` is gravitational acceleration, `α` is the thermal expansion
coefficient, `β` is the haline contraction coefficient, `Θ` is
conservative temperature, and `sᴬ` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that `∂y_Θ`, `∂y_sᴬ`, `α`, and `β` are all evaluated at cell interfaces in `y`
and cell centers in `x` and `z`.
"""
@inline function ∂y_b(i, j, k, grid, b::SeawaterBuoyancy, C)
    Θ, sᴬ = get_temperature_and_salinity(b, C)
    return b.gravitational_acceleration * (
           thermal_expansionᶜᶠᶜ(i, j, k, grid, b.equation_of_state, Θ, sᴬ) * ∂yᵃᶠᵃ(i, j, k, grid, Θ)
        - haline_contractionᶜᶠᶜ(i, j, k, grid, b.equation_of_state, Θ, sᴬ) * ∂yᵃᶠᵃ(i, j, k, grid, sᴬ) )
end

"""
    ∂z_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the vertical derivative of buoyancy for temperature and salt-stratified water,

```math
∂_z b = N^2 = g ( α ∂_z Θ - β ∂_z sᴬ ) ,
```

where `g` is gravitational acceleration, `α` is the thermal expansion
coefficient, `β` is the haline contraction coefficient, `Θ` is
conservative temperature, and `sᴬ` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that `∂z_Θ`, `∂z_sᴬ`, `α`, and `β` are all evaluated at cell interfaces in `z`
and cell centers in `x` and `y`.
"""
@inline function ∂z_b(i, j, k, grid, b::SeawaterBuoyancy, C)
    Θ, sᴬ = get_temperature_and_salinity(b, C)
    return b.gravitational_acceleration * (
           thermal_expansionᶜᶜᶠ(i, j, k, grid, b.equation_of_state, Θ, sᴬ) * ∂zᵃᵃᶠ(i, j, k, grid, Θ)
        - haline_contractionᶜᶜᶠ(i, j, k, grid, b.equation_of_state, Θ, sᴬ) * ∂zᵃᵃᶠ(i, j, k, grid, sᴬ) )
end

@inline function buoyancy_perturbation(i, j, k, grid, b::SeawaterBuoyancy{FT, <:AbstractNonlinearEquationOfState}, C) where FT
    Θ, sᴬ = get_temperature_and_salinity(b, C)
    return - b.gravitational_acceleration * ρ′(i, j, k, grid, b.equation_of_state, Θ, sᴬ) / b.equation_of_state.ρ₀
end
