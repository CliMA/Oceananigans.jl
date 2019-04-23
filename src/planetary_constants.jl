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

function Europa(; f=nothing, lat=nothing)
    # Values taken from Soderlund (Table 1, 2019) [arXiv:1901.04093].
    Ω = 2.1e-5
    g = 1.3

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

    PlanetaryConstants(Ω, f, g)
end

function Enceladus(; f=nothing, lat=nothing)
    # Values taken from Soderlund (Table 1, 2019) [arXiv:1901.04093].
    Ω = 5.3e-5
    g = 0.1

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

    PlanetaryConstants(Ω, f, g)
end
