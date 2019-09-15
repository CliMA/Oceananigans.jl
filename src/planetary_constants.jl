struct PlanetaryConstants{T<:AbstractFloat}
    Ω::T  # Rotation rate of the planet [rad/s].
    f::T  # Nominal value for the Coriolis frequency [rad/s].
    g::T  # Standard acceleration due to gravity [m/s²].
end

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

PlanetaryConstants(T=Float64; Ω, g, f=nothing, latitude=nothing) =
    PlanetaryConstants{T}(Ω, choose_f(Ω, f, latitude), g)

PlanetaryConstants(T=Float64; f, g) = PlanetaryConstants{T}(0, f, g)

const Ω_Earth = 7.2921150e-5
const g_Earth = 9.80665

function Earth(T=Float64; f=nothing, latitude=nothing)
    isnothing(f) && isnothing(latitude) && (f = 1e-4)
    return PlanetaryConstants{T}(Ω_Earth, choose_f(Ω_Earth, f, latitude), g_Earth)
end
