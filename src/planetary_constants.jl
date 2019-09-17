struct PlanetaryConstants{T<:AbstractFloat}
    g :: T # Standard acceleration due to gravity [m/s²].
end

PlanetaryConstants(T::DataType=Float64; g) = PlanetaryConstants{T}(g)

const Ω_Earth = 7.2921150e-5
const g_Earth = 9.80665

Earth(T=Float64) = PlanetaryConstants{T}(g_Earth)
