"""
    PlanetaryConstants{T<:AbstractFloat}

A type that stores the rotation rate ``Ω``, Coriolis frequency ``f``, and gravitational
acceleration ``g`` experiences by the fluid. Constants are stored as floating point
numbers of type `T`.
"""
struct PlanetaryConstants{T<:AbstractFloat}
    Ω::T  # Rotation rate of the planet [rad/s].
    f::T  # Nominal value for the Coriolis frequency [rad/s].
    g::T  # Standard acceleration due to gravity [m/s²].
end

"""
    choose_f(Ω, f, lat)

Utility for choosing a value for the Coriolis frequency ``f`` given a rotation rate ``Ω``
and a latitude `lat`. Used by constructors that allow choosing an ``f`` or a `lat`.
"""
function choose_f(Ω, f, lat)
    if isnothing(f) && isnothing(lat)
        throw(ArgumentError("Must specify Coriolis parameter f or latitude!"))
    elseif isnothing(lat)
        f′ = f
    elseif isnothing(f)
        f′ = 2*Ω*sind(lat)
    else
        throw(ArgumentError("Cannot specify both f and lat!"))
    end

    abs(f′) <= 2Ω || throw(ArgumentError("Coriolis parameter |f| cannot be larger than 2Ω!"))

    return f′
end

"""
    PlanetaryConstants(T=Float64; Ω, g, f=nothing, latitude=nothing)

Constructs a `PlanetaryConstants` type given a rotation rate ``Ω``, gravitational
acceleration ``g``, and a Coriolis frequency ``f`` or a latitude `lat`.
"""
PlanetaryConstants(T=Float64; Ω, g, f=nothing, latitude=nothing) =
    PlanetaryConstants{T}(Ω, choose_f(Ω, f, latitude), g)

"""
    PlanetaryConstants(T=Float64; f, g)

Constructs a `PlanetaryConstants` type given a Coriolis frequency ``f`` and a gravitational
acceleration ``g``.
"""
PlanetaryConstants(T=Float64; f, g) = PlanetaryConstants{T}(0, f, g)

# Taken from https://en.wikipedia.org/wiki/Earth%27s_rotation#Angular_speed
# where it is given as 7.2921150 ± 0.0000001×10⁻⁵ radians per SI second.
const Ω_Earth = 7.292115e-5

# Value used is the standard acceleration due to gravity:
# https://en.wikipedia.org/wiki/Standard_gravity
const g_Earth = 9.80665

"""
    Earth(T=Float64; f=nothing, latitude=nothing)

Construct a `PlanetaryConstants` type given a Coriolis frequency ``f`` or a latitude `lat`
assuming the Earth's rotation rate ``Ω`` and standard gravitational acceleration on Earth.

For legacy purposes, if no ``f`` and no `lat` are given, a default value of ``f = 10^{-4}``
s⁻¹ is used.
"""
function Earth(T=Float64; f=nothing, latitude=nothing)
    isnothing(f) && isnothing(latitude) && (f = 1e-4)
    return PlanetaryConstants{T}(Ω_Earth, choose_f(Ω_Earth, f, latitude), g_Earth)
end
