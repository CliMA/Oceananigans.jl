using JULES, Oceananigans
using OffsetArrays, Test

Nx = Ny = Nz = 8
L = 100

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(0, L), y=(0, L), z=(0, L))

const pₛ = 100000
const Tₐ = 293.15
const g  = 9.80665
const Γ  = -9.8e-3

buoyancy = IdealGas()
const Rᵈ = buoyancy.Rᵈ
const γ  = buoyancy.γ
const ρₛ = pₛ / (Rᵈ*Tₐ)

# Isothermal atmosphere
Tₚ(z) = Tₐ + Γ*z
p₀(x, y, z) = pₛ * (Tₐ / Tₚ(z))^(g / (Rᵈ*Γ))
ρ₀(x, y, z) = ρₛ * (Tₐ / Tₚ(z))^(1 + g /(Rᵈ*Γ))
T₀(x, y, z) = ρ₀(x, y, z) * Tₚ(z)

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

@test all(isapprox.(model.momenta.W.data, 0, atol=2e-9))

