struct PlanetaryConstants{T<:AbstractFloat} <: ConstantsCollection
    Ω::T  # Rotation rate of the planet [rad/s].
    f::T  # Nominal value for the Coriolis frequency [rad/s].
    g::T  # Standard acceleration due to gravity [m/s²].
end

function choose_f(Ω, f, lat)
end

function Earth(; f=nothing, lat=nothing)
    Ω = 7.2921150e-5
    g = 9.80665

    if isnothing(f) && isnothing(lat)
        f′ = 1e-4  # Corresponds to a latitude of 43.29°N.
    elseif isnothing(lat)
        f′ = f
    elseif isnothing(f)
        f′ = 2*Ω*sind(lat)
    else
        throw(ArgumentError("Cannot specify both f and lat!"))
    end

    abs(f′) <= 2Ω || throw(ArgumentError("Coriolis parameter |f| cannot be larger than 2Ω!"))

    PlanetaryConstants(Ω, f′, g)
end

function EarthStationary()
    Ω = 0
    f = 0
    g = 9.80665
    PlanetaryConstants(Ω, f, g)
end
