using JULES, Oceananigans
using Plots, Printf
using OffsetArrays

Nx = Ny = Nz = 8
L = 10e3

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(0, L), y=(0, L), z=(0, L))

pₛ = 100000
Tₐ = 293.15
g  = 9.80665

buoyancy = IdealGas()
Rᵈ = buoyancy.Rᵈ
cₚ = buoyancy.cₚ
ρₛ = pₛ / (Rᵈ*Tₐ)

# Isothermal atmosphere
H = Rᵈ * Tₐ / g
p₀(x, y, z) = pₛ * exp(-z/H)
ρ₀(x, y, z) = ρₛ * exp(-z/H)

θ₀(x, y, z) = Tₐ * exp(z/H * Rᵈ/cₚ)
Θ₀(x, y, z) = ρ₀(x, y, z) * θ₀(x, y, z)

model = CompressibleModel(grid=grid, buoyancy=buoyancy, surface_pressure=pₛ)

set!(model.density, ρ₀)
set!(model.tracers.Θᵐ, Θ₀)

Δtp = 1e-3
time_step!(model; Δt=Δtp, nₛ=1)
