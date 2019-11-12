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

# Isothermal atmosphere
p₀(x, y, z) = pₛ * exp(- g*z / (Rᵈ*Tₐ))
ρ₀(x, y, z) = ρₛ * exp(- g*z / (Rᵈ*Tₐ))
T₀(x, y, z) = ρ₀(x, y, z) * Tₐ

zC = grid.zC
p₀a = @. p₀(0, 0, zC)
p₀oa = OffsetArray([p₀a[1]; p₀a; p₀a[end]], 0:Nz+1)

base_state = BaseState(p=p₀oa, ρ=ρ₀)

model = CompressibleModel(grid=grid, buoyancy=buoyancy, surface_pressure=pₛ,
                          prognostic_temperature=Temperature(), base_state=base_state)

set!(model.density, ρ₀)
set!(model.tracers.Θᵐ, T₀)

Δtp = 1e-3
time_step!(model; Δt=Δtp, nₛ=1)

