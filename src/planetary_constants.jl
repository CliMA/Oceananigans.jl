struct PlanetaryConstants <: ConstantsCollection
    Ω::Float64  # Rotation rate of the planet [rad/s].
    f::Float64  # Nominal value for the Coriolis frequency [rad/s].
    g::Float64  # Standard acceleration due to gravity [m/s²].
end

function Earth()
    Ω = 7.2921150e-5
    f = 1e-4
    g = 9.80665
    PlanetaryConstants(Ω, f, g)
end

function EarthStationary()
    Ω = 0
    f = 0
    g = 9.80665
    PlanetaryConstants(Ω, f, g)
end
