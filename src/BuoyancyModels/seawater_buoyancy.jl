using Oceananigans.BoundaryConditions: NoFluxBoundaryCondition
using Oceananigans.Utils: prettysummary

"""
    SeawaterBuoyancy{FT, EOS, T, S} <: AbstractBuoyancyModel{EOS}

BuoyancyModels model for seawater. `T` and `S` are either `nothing` if both
temperature and salinity are active, or of type `FT` if temperature
or salinity are constant, respectively.
"""
struct SeawaterBuoyancy{FT, EOS, T, S} <: AbstractBuoyancyModel{EOS}
             equation_of_state :: EOS
    gravitational_acceleration :: FT
          constant_temperature :: T
             constant_salinity :: S
end

required_tracers(::SeawaterBuoyancy) = (:T, :S)
required_tracers(::SeawaterBuoyancy{FT, EOS, <:Nothing, <:Number}) where {FT, EOS} = (:T,) # active temperature only
required_tracers(::SeawaterBuoyancy{FT, EOS, <:Number, <:Nothing}) where {FT, EOS} = (:S,) # active salinity only

Base.nameof(::Type{SeawaterBuoyancy}) = "SeawaterBuoyancy"
Base.summary(b::SeawaterBuoyancy) = string(nameof(typeof(b)), " with g=", prettysummary(b.gravitational_acceleration),
                                           " and ", summary(b.equation_of_state))

function Base.show(io::IO, b::SeawaterBuoyancy{FT}) where FT

    print(io, nameof(typeof(b)), "{$FT}:", "\n",
              "├── gravitational_acceleration: ", b.gravitational_acceleration, "\n")

    if !isnothing(b.constant_temperature)
        print(io, "├── constant_temperature: ", b.constant_temperature, "\n")
    end

    if !isnothing(b.constant_salinity)
        print(io, "├── constant_salinity: ", b.constant_salinity, "\n")
    end
        
    print(io, "└── equation_of_state: ", summary(b.equation_of_state))
end

"""
    SeawaterBuoyancy([FT = Float64;]
                     gravitational_acceleration = g_Earth,
                     equation_of_state = LinearEquationOfState(FT),
                     constant_temperature = nothing,
                     constant_salinity = nothing)

Return parameters for a temperature- and salt-stratified seawater buoyancy model
with a `gravitational_acceleration` constant (typically called ``g``), and an
`equation_of_state` that related temperature and salinity (or conservative temperature
and absolute salinity) to density anomalies and buoyancy.

Setting `constant_temperature` to something that is not `nothing` indicates that buoyancy depends only on salinity.
For a nonlinear equation of state, the value provided `constant_temperature` is used as the temperature of the system.
Vice versa, setting `constant_salinity` indicates that buoyancy depends only on temperature.

For a linear equation of state, the values of `constant_temperature` or `constant_salinity`
are irrelevant.

Examples
========

The "TEOS10" equation of state, see https://www.teos-10.org

```jldoctest seawaterbuoyancy
julia> using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

julia> teos10 = TEOS10EquationOfState()
BoussinesqEquationOfState{Float64}:
    ├── seawater_polynomial: TEOS10SeawaterPolynomial{Float64}
    └── reference_density: 1020.0
```

Buoyancy that depends on both temperature and salinity

```jldoctest seawaterbuoyancy
julia> using Oceananigans

julia> buoyancy = SeawaterBuoyancy(equation_of_state=teos10)
SeawaterBuoyancy{Float64}:
├── gravitational_acceleration: 9.80665
└── equation_of_state: BoussinesqEquationOfState{Float64}
```

Buoyancy that depends only on salinity with temperature held at 20 degrees Celsius

```jldoctest seawaterbuoyancy
julia> salinity_dependent_buoyancy = SeawaterBuoyancy(equation_of_state=teos10, constant_temperature=20) 
SeawaterBuoyancy{Float64}:
├── gravitational_acceleration: 9.80665
├── constant_temperature: 20
└── equation_of_state: BoussinesqEquationOfState{Float64}
```

Buoyancy that depends only on temperature with salinity held at 35 psu

```jldoctest seawaterbuoyancy
julia> temperature_dependent_buoyancy = SeawaterBuoyancy(equation_of_state=teos10, constant_salinity=35)
SeawaterBuoyancy{Float64}:
├── gravitational_acceleration: 9.80665
├── constant_salinity: 35
└── equation_of_state: BoussinesqEquationOfState{Float64}
```
"""
function SeawaterBuoyancy(FT = Float64;
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
                            equation_of_state, gravitational_acceleration, constant_temperature, constant_salinity)
end

const TemperatureSeawaterBuoyancy = SeawaterBuoyancy{FT, EOS, <:Nothing, <:Number} where {FT, EOS}
const SalinitySeawaterBuoyancy = SeawaterBuoyancy{FT, EOS, <:Number, <:Nothing} where {FT, EOS}

Base.nameof(::Type{TemperatureSeawaterBuoyancy}) = "TemperatureSeawaterBuoyancy"
Base.nameof(::Type{SalinitySeawaterBuoyancy}) = "SalinitySeawaterBuoyancy"

@inline get_temperature_and_salinity(::SeawaterBuoyancy, C) = C.T, C.S
@inline get_temperature_and_salinity(b::TemperatureSeawaterBuoyancy, C) = C.T, b.constant_salinity
@inline get_temperature_and_salinity(b::SalinitySeawaterBuoyancy, C) = b.constant_temperature, C.S

@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, b::SeawaterBuoyancy, C)
    T, S = get_temperature_and_salinity(b, C)
    return - (b.gravitational_acceleration * ρ′(i, j, k, grid, b.equation_of_state, T, S)
              / b.equation_of_state.reference_density)
end

#####
##### Buoyancy gradient components
#####

"""
    ∂x_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the ``x``-derivative of buoyancy for temperature and salt-stratified water,

```math
∂_x b = g ( α ∂_x T - β ∂_x S ) ,
```

where ``g`` is gravitational acceleration, ``α`` is the thermal expansion
coefficient, ``β`` is the haline contraction coefficient, ``T`` is
conservative temperature, and ``S`` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that ``∂_x T`` (`∂x_T`), ``∂_x S`` (`∂x_S`), ``α``, and ``β`` are all evaluated at cell
interfaces in `x` and cell centers in `y` and `z`.
"""
@inline function ∂x_b(i, j, k, grid, b::SeawaterBuoyancy, C)
    T, S = get_temperature_and_salinity(b, C)
    return b.gravitational_acceleration * (
           thermal_expansionᶠᶜᶜ(i, j, k, grid, b.equation_of_state, T, S) * ∂xᶠᶜᶜ(i, j, k, grid, T)
        - haline_contractionᶠᶜᶜ(i, j, k, grid, b.equation_of_state, T, S) * ∂xᶠᶜᶜ(i, j, k, grid, S) )
end

"""
    ∂y_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the ``y``-derivative of buoyancy for temperature and salt-stratified water,

```math
∂_y b = g ( α ∂_y T - β ∂_y S ) ,
```

where ``g`` is gravitational acceleration, ``α`` is the thermal expansion
coefficient, ``β`` is the haline contraction coefficient, ``T`` is
conservative temperature, and ``S`` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that ``∂_y T`` (`∂y_T`), ``∂_y S`` (`∂y_S`), ``α``, and ``β`` are all evaluated at cell
interfaces in `y` and cell centers in `x` and `z`.
"""
@inline function ∂y_b(i, j, k, grid, b::SeawaterBuoyancy, C)
    T, S = get_temperature_and_salinity(b, C)
    return b.gravitational_acceleration * (
           thermal_expansionᶜᶠᶜ(i, j, k, grid, b.equation_of_state, T, S) * ∂yᶜᶠᶜ(i, j, k, grid, T)
        - haline_contractionᶜᶠᶜ(i, j, k, grid, b.equation_of_state, T, S) * ∂yᶜᶠᶜ(i, j, k, grid, S) )
end

const f = Face()
const c = Center()

"""
    ∂z_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the vertical derivative of buoyancy for temperature and salt-stratified water,

```math
∂_z b = N^2 = g ( α ∂_z T - β ∂_z S ) ,
```

where ``g`` is gravitational acceleration, ``α`` is the thermal expansion
coefficient, ``β`` is the haline contraction coefficient, ``T`` is
conservative temperature, and ``S`` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that ``∂_z T`` (`∂z_T`), ``∂_z S`` (`∂z_S`), ``α``, and ``β`` are all evaluated at cell
interfaces in `z` and cell centers in `x` and `y`.
"""
@inline function ∂z_b(i, j, k, grid, b::SeawaterBuoyancy, C)
    T, S = get_temperature_and_salinity(b, C)
    g = b.gravitational_acceleration

    α = thermal_expansionᶜᶜᶠ(i, j, k, grid, b.equation_of_state, T, S)
    β = haline_contractionᶜᶜᶠ(i, j, k, grid, b.equation_of_state, T, S)

    return g * (α * ∂zᶜᶜᶠ(i, j, k, grid, T) - β * ∂zᶜᶜᶠ(i, j, k, grid, S))
end

#####
##### top buoyancy flux
#####

@inline get_temperature_and_salinity_flux(::SeawaterBuoyancy, bcs) = bcs.T, bcs.S
@inline get_temperature_and_salinity_flux(::TemperatureSeawaterBuoyancy, bcs) = bcs.T, NoFluxBoundaryCondition()
@inline get_temperature_and_salinity_flux(::SalinitySeawaterBuoyancy, bcs) = NoFluxBoundaryCondition(), bcs.S

@inline function top_bottom_buoyancy_flux(i, j, k, grid, b::SeawaterBuoyancy, top_bottom_tracer_bcs, clock, fields)
    T, S = get_temperature_and_salinity(b, fields)
    T_flux_bc, S_flux_bc = get_temperature_and_salinity_flux(b, top_bottom_tracer_bcs)

    T_flux = getbc(T_flux_bc, i, j, grid, clock, fields)
    S_flux = getbc(S_flux_bc, i, j, grid, clock, fields)

    return b.gravitational_acceleration * (
              thermal_expansionᶜᶜᶠ(i, j, k, grid, b.equation_of_state, T, S) * T_flux
           - haline_contractionᶜᶜᶠ(i, j, k, grid, b.equation_of_state, T, S) * S_flux)
end

@inline    top_buoyancy_flux(i, j, grid, b::SeawaterBuoyancy, args...) = top_bottom_buoyancy_flux(i, j, grid.Nz+1, grid, b, args...)
@inline bottom_buoyancy_flux(i, j, grid, b::SeawaterBuoyancy, args...) = top_bottom_buoyancy_flux(i, j, 1, grid, b, args...)

