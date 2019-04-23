struct PlanetaryConstants{T<:AbstractFloat} <: ConstantsCollection
    Ω::T  # Rotation rate of the planet [rad/s].
    f::T  # Nominal value for the Coriolis frequency [rad/s].
    g::T  # Standard acceleration due to gravity [m/s²].
end

function Earth(lat=nothing)
    Ω = 7.2921150e-5

    if isnothing(lat)
        f = 1e-4  # Corresponds to a latitude of 43.29°N.
    else
        f = 2*Ω*sind(lat)
    end

    g = 9.80665
    PlanetaryConstants(Ω, f, g)
end

function EarthStationary()
    Ω = 0
    f = 0
    g = 9.80665
    PlanetaryConstants(Ω, f, g)
end
