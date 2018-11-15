using Statistics: mean

using OceanDispatch

function generate_cooling_disk!(c::PlanetaryConstants, g::Grid, Fũ::ForcingFields)
    # Parameters for generating initial surface heat flux.
    Rc = 600  # Radius of cooling disk [m].
    Ts = 20  # Surface temperature [°C].
    Q₀ = 800  # Cooling disk heat flux [W/m²].
    Q₁ = 10  # Noise added to cooling disk heat flux [W/m²].
    Ns = 0 * (c.f * Rc/g.Lz)  # Stratification or Brunt–Väisälä frequency [s⁻¹].

    αᵥ = 2.07e-4  # Volumetric coefficient of thermal expansion for water [K⁻¹].
    cᵥ = 4181.3   # Isobaric mass heat capacity [J / kg·K].

    Tz = Ns^2 / (c.g * αᵥ)  # Vertical temperature gradient [K/m].

    # Center horizontal coordinates so that (x,y) = (0,0) corresponds to the center
    # of the domain (and the cooling disk).
    x₀ = g.xCA[:, :, 1] .- mean(g.xCA[:, :, 1])
    y₀ = g.yCA[:, :, 1] .- mean(g.yCA[:, :, 1])

    # Calculate vertical temperature profile and convert to Kelvin.
    T_ref = 273.15 .+ Ts .+ Tz .* (g.zCR .- mean(Tz * g.zCR))

    # Set surface heat flux to zero outside of cooling disk of radius Rᶜ.
    r₀² = x₀.*x₀ + y₀.*y₀

    # Generate surface heat flux field with small random fluctuations.
    Q = Q₀ .+ Q₁ * (0.5 .+ rand(g.Nx, g.Ny))
    Q[findall(r₀² .> Rc^2)] .= 0  # Set Q=0 outside cooling disk.

    ρ₀ = 1.027e3  # Reference density [kg/m³]

    # Convert surface heat flux into 3D forcing term for use when calculating
    # source terms at each time step. Also convert surface heat flux [W/m²]
    # into a temperature tendency forcing [K/s].
    @. Fũ.Fθ.f[:, :, 1] = (Q / cᵥ) * (g.Az / (ρ₀ * g.V))
    nothing
end

function impose_initial_conditions!(c::PlanetaryConstants, g::Grid, fs::Fields, Fũ::ForcingFields)
    generate_cooling_disk!(c, g, Fũ)

    ρ₀ = 1.027e3  # Reference density [kg/m³]

    fs.u.f .= 0; fs.v.f .= 0; fs.w.f .= 0;
    fs.S.f .= 35;  # TODO: Should set to EOS.S₀
    fs.T.f .= 273.15 + 20;  # 20°C.

    pHY_profile = [-ρ₀ * c.g * h for h in g.zCR]
    fs.pHY.f .= repeat(reshape(pHY_profile, 1, 1, g.Nz), g.Nx, g.Ny, 1)
    fs.p.f .= fs.pHY.f  # Initial pressure is just the hydrostatic pressure.

    # fs.ρ .= ρ!(eos, fs.θ, fs.S, fs.p)
    nothing
end

c = EarthConstants()

g =  RegularCartesianGrid((100, 100, 50), (2000, 2000, 1000), Float64)
fs = Fields(g)
Gũ = SourceTermFields(g)
Fũ = ForcingFields(g)

impose_initial_conditions!(c, g, fs, Fũ)
