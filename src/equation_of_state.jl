#=
Linear equation of state for seawater. Constants taken from Table 1.2 (page 33)
and functional form taken from Eq. (1.57) of Vallis, "Atmospheric and Oceanic
Fluid Dynamics: Fundamentals and Large-Scale Circulation" (2ed, 2017). Note
that a linear equation of state is not accurate enough for serious quantitative
oceanography as the expansion and contraction β coefficients vary with
temperature, pressure, and salinity.
=#

struct LinearEquationOfState <: EquationOfStateParameters
    ρ₀::Float64  # Reference density [kg/m³]
    βT::Float64  # First thermal expansion coefficient [1/K]
    βS::Float64  # Haline contraction coefficient [1/ppt]
    βp::Float64  # Compressibility coefficient [ms²/kg]
    T₀::Float64  # Reference temperature [K]
    S₀::Float64  # Reference salinity [g/kg]
    p₀::Float64  # Reference pressure [Pa].
    cᵥ::Float64  # Isobaric mass heat capacity [J / kg·K].
    αᵥ::Float64  # Volumetric coefficient of thermal expansion for water [K⁻¹].
end

function LinearEquationOfState()
    ρ₀ = 1.027e3
    βT = 1.67e-4
    βS = 0.78e-3
    βp = 4.39e-10
    T₀ = 283
    S₀ = 35
    p₀ = 1e5
    cᵥ = 4181.3
    αᵥ = 2.07e-4
    LinearEquationOfState(ρ₀, βT, βS, βp, T₀, S₀, p₀, cᵥ, αᵥ)
end

function ρ!(s::LinearEquationOfState, g::Grid, tr::TracerFields)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds tr.ρ.data[i, j, k] =  s.ρ₀ * (1 - s.βT * (tr.T.data[i, j, k] - s.T₀))
    end
    nothing
end

function δρ!(s::LinearEquationOfState, g::Grid, δρ::CellField, T::CellField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds δρ.data[i, j, k] =  - s.ρ₀ * s.βT * (T.data[i, j, k] - s.T₀)
    end
    nothing
end

function ∫δρgdz!(g::Grid, c::PlanetaryConstants, δρ::CellField, δρz::FaceFieldZ, pHY′::CellField)
    gΔz = c.g * g.Δz
    for j in 1:g.Ny, i in 1:g.Nx
      pHY′.data[i, j, 1] = δρ.data[i, j, 1] * gΔz / 2
    end
    for k in 2:g.Nz, j in 1:g.Ny, i in 1:g.Nx
      pHY′.data[i, j, k] = pHY′.data[i, j, k-1] + (δρz.data[i, j, k] * gΔz)
    end
end
