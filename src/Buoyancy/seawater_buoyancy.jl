"""
    SeawaterBuoyancy{G, EOS} <: AbstractBuoyancy{EOS}

Buoyancy model for temperature- and salt-stratified seawater.
"""
struct SeawaterBuoyancy{FT, EOS} <: AbstractBuoyancy{EOS}
    gravitational_acceleration :: FT
    equation_of_state :: EOS
end

required_tracers(::SeawaterBuoyancy) = (:T, :S)

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
           thermal_expansionᶠᶜᶜ(i, j, k, grid, b.equation_of_state, C) * ∂xᶠᵃᵃ(i, j, k, grid, C.T)
        - haline_contractionᶠᶜᶜ(i, j, k, grid, b.equation_of_state, C) * ∂xᶠᵃᵃ(i, j, k, grid, C.S) )

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
           thermal_expansionᶜᶠᶜ(i, j, k, grid, b.equation_of_state, C) * ∂yᵃᶠᵃ(i, j, k, grid, C.T)
        - haline_contractionᶜᶠᶜ(i, j, k, grid, b.equation_of_state, C) * ∂yᵃᶠᵃ(i, j, k, grid, C.S) )


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
           thermal_expansionᶜᶜᶠ(i, j, k, grid, b.equation_of_state, C) * ∂zᵃᵃᶠ(i, j, k, grid, C.T)
        - haline_contractionᶜᶜᶠ(i, j, k, grid, b.equation_of_state, C) * ∂zᵃᵃᶠ(i, j, k, grid, C.S) )
