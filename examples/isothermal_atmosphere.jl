using JULES, Oceananigans
using Plots, Printf
using OffsetArrays

Nx = Ny = Nz = 8
L = 100

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(0, L), y=(0, L), z=(0, L))

const pₛ = 100000
const Tₐ = 293.15
const g  = 9.80665

buoyancy = IdealGas()
const Rᵈ = buoyancy.Rᵈ
const γ  = buoyancy.γ
const ρₛ = pₛ / (Rᵈ*Tₐ)

model = CompressibleModel(grid=grid, buoyancy=buoyancy, surface_pressure=pₛ)

# Isothermal atmosphere
p₀(x, y, z) = pₛ * exp(- g*z / (Rᵈ*Tₐ))
ρ₀(x, y, z) = ρₛ * exp(- g*z / (Rᵈ*Tₐ))
Θ₀(x, y, z) = pₛ/Rᵈ * (pₛ / p₀(x, y, z))^γ

set!(model.density, ρ₀)
set!(model.tracers.Θᵐ, Θ₀)

Δtp = 1e-3
time_step!(model; Δt=Δtp, nₛ=1)

