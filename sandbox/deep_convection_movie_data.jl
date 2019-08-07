using Statistics, Printf
using Oceananigans

# Simulation parameters
FT = Float64
Nx, Ny, Nz = 256, 256, 256     # Number of grid points.
Lx, Ly, Lz = 2000, 2000, 1000  # Domain size [meters].
end_time = 4day
Δt = 4  # Time step in seconds.

grid = RegularCartesianGrid(FT, (Nx, Ny, Nz), (Lx, Ly, Lz))

# Physical constants
φ  = 58       # Latitude to run simulation at. Corresponds to Labrador Sea.
ρ₀ = 1027    # Density of seawater [kg/m³]
αᵥ = 2.07e-4  # Volumetric coefficient of thermal expansion for water [K⁻¹].
cₚ = 4181.3   # Isobaric mass heat capacity [J / kg·K].

planetary_constants = Earth(lat=φ)
f, g = planetary_constants.f, planetary_constants.g  # Coriolis parameter and standard gravity.

# Parameters for generating initial surface heat flux.
Rc = 600   # Radius of cooling disk [m].
Ts = 20    # Surface temperature [°C].
Q₀ = -800  # Cooling disk heat flux [W/m²].
Q₁ = 1     # Noise added to cooling disk heat flux [W/m²].
Ns = 5 * (f * Rc/Lz)  # Stratification or Brunt–Väisälä frequency [s⁻¹].
∂T∂z = Ns^2 / (g * αᵥ)  # Vertical temperature gradient [K/m].

# Center horizontal coordinates so that (x₀,y₀) = (0,0) corresponds to the center
# of the domain (and the cooling disk).
x₀ = grid.xC .- mean(grid.xC)
y₀ = grid.yC .- mean(grid.yC)

# Generate surface heat flux field.
Q = (Q₀ .+ Q₁ .* (0.5 .+ randn(Nx, Ny))) / (ρ₀ * cₚ)

if HAVE_CUDA
    using CuArrays
    Q = CuArray(Q)
end

# Set surface heat flux to zero outside of cooling disk of radius Rc.
r₀² = @. x₀*x₀ + y₀'*y₀'
Q[findall(r₀² .> Rc^2)] .= 0

Tbcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Flux, Q),
                                bottom = BoundaryCondition(Gradient, ∂T∂z))

model = Model(float_type = FT,
                    arch = HAVE_CUDA ? GPU() : CPU(),
                       N = (Nx, Ny, Nz),
                       L = (Lx, Ly, Lz),
                       ν = 4e-2, κ = 4e-2,
               constants = planetary_constants,
                     bcs = BoundaryConditions(T=Tbcs))

ε(μ) = μ * randn()  # noise
T₀(x, y, z) = Ts + ∂T∂z * z + ε(1e-4)

set_ic!(model; T=T₀)

# Only save the temperature field (excluding halo regions).
Hx, Hy, Hz = model.grid.Hx, model.grid.Hy, model.grid.Hz
θ(model) = Array(model.tracers.T.data.parent[1+Hx:Nx+Hx, 1+Hy:Ny+Hy, 1+Hz:Nz+Hz])
fields = Dict(:θ=>θ)

# Save output every 100 seconds.
field_writer = JLD2OutputWriter(model, fields; dir="data", prefix="deep_convection_temperatre", interval=100, force=true)

push!(model.output_writers, field_writer)

Ni = 1  # Number of "intermediate" time steps to take before printing a progress message.
while model.clock.time < end_time
    walltime = @elapsed time_step!(model; Nt=Ni, Δt=Δt)

    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)
    CFL = Δt / cell_advection_timescale(model)

    progress = 100 * (model.clock.time / end_time)
    @printf("[%06.2f%%] i: %d, t: %8.5g, umax: (%6.3g, %6.3g, %6.3g), CFL: %6.4g, ⟨wall time⟩: %s\n",
            progress, model.clock.iteration, model.clock.time, umax, vmax, wmax, CFL, prettytime(1e9*walltime / Ni))
end
