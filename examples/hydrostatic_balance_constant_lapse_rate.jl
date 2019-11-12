using JULES, Oceananigans
using Plots, Printf
using OffsetArrays

Nx = Ny = Nz = 8
L = 100

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(0, L), y=(0, L), z=(0, L))

buoyancy = IdealGas()
const g  = 9.80665
const Rᵈ = buoyancy.Rᵈ

const pₛ = 100000
const Tₛ = 293.15
const Γ  = -9.8e-3  # K/m
const ρₛ = pₛ / (Rᵈ*Tₛ)

model = CompressibleModel(grid=grid, buoyancy=buoyancy, surface_pressure=pₛ,
                          prognostic_temperature=Temperature())

T₀(x, y, z) = Tₛ + Γ*z
p₀(x, y, z) = pₛ * (Tₛ / T₀(x, y, z))^(g/(Rᵈ*Γ))
ρ₀(x, y, z) = ρₛ * (Tₛ / T₀(x, y, z))^(1 + g/(Rᵈ*Γ))

set!(model.density, ρ₀)
set!(model.tracers.Θᵐ, T₀)

Δtp = 1e-3
time_step!(model; Δt=Δtp, nₛ=1)

