# using Plots
using Statistics
using CuArrays, CUDAnative, GPUifyLoops
using Oceananigans

function deep_convection_3d_gpu()
    Nx, Ny, Nz = 128, 128, 64
    Lx, Ly, Lz = 2000, 2000, 1000
    Nt, Δt = 1000, 20

    model = Model((Nx, Ny, Nz), (Lx, Ly, Lz), :gpu, Float32)
    impose_initial_conditions!(model)

    field_writer = FieldWriter(".", "deep_convection_3d_gpu", 10, [model.tracers.T], ["T"])
    push!(model.output_writers, field_writer)
    
    time_step_kernel!(model, Nt, Δt)
end

function impose_cooling_disk!(model::Model)
    g = model.grid
    c = model.constants

    # Parameters for generating initial surface heat flux.
    Rc = 600  # Radius of cooling disk [m].
    Ts = 20  # Surface temperature [°C].
    Q₀ = -800  # Cooling disk heat flux [W/m²].
    Q₁ = 10  # Noise added to cooling disk heat flux [W/m²].
    Ns = 5 * (c.f * Rc/g.Lz)  # Stratification or Brunt–Väisälä frequency [s⁻¹].

    αᵥ = 2.07e-4  # Volumetric coefficient of thermal expansion for water [K⁻¹].
    cᵥ = 4181.3   # Isobaric mass heat capacity [J / kg·K].

    Tz = Ns^2 / (c.g * αᵥ)  # Vertical temperature gradient [K/m].

    # Center horizontal coordinates so that (x,y) = (0,0) corresponds to the center
    # of the domain (and the cooling disk).
    x₀ = g.xC .- mean(g.xC)
    y₀ = g.yC .- mean(g.yC)

    # Calculate vertical temperature profile and convert to Kelvin.
    T_ref = 273.15 .+ Ts .+ Tz .* (g.zC .- mean(Tz * g.zC))

    # Impose reference temperature profile.
    model.tracers.T.data .= cu(repeat(reshape(T_ref, 1, 1, g.Nz), g.Nx, g.Ny, 1))

    # Set surface heat flux to zero outside of cooling disk of radius Rᶜ.
    r₀² = @. x₀*x₀ + y₀'*y₀'

    # Generate surface heat flux field with small random fluctuations.
    Q = Q₀ .+ Q₁ * (0.5 .+ rand(g.Nx, g.Ny))
    Q[findall(r₀² .> Rc^2)] .= 0  # Set Q=0 outside cooling disk.

    # Convert surface heat flux into 3D forcing term for use when calculating
    # source terms at each time step. Also convert surface heat flux [W/m²]
    # into a temperature tendency forcing [K/s].
    T_forcing = zeros(g.Nx, g.Ny, g.Nz)
    T_forcing[:, :, 1] = (Q / cᵥ) * (g.Az / (model.eos.ρ₀ * g.V))
    model.forcings.FT.data .= cu(T_forcing)
    nothing
end

function impose_initial_conditions!(model::Model)
    g = model.grid

    impose_cooling_disk!(model)

    @. model.tracers.S.data = model.eos.S₀

    pHY_profile = [-model.eos.ρ₀ * model.constants.g * h for h in g.zC]
    model.pressures.pHY.data .= cu(repeat(reshape(pHY_profile, 1, 1, g.Nz), g.Nx, g.Ny, 1))
    nothing
end