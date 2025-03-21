using Oceananigans
using Oceananigans.Units

arch = CPU()
Nx = Ny = Nz = 64
Lx = Ly = 512
Lz = 256
grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(0, Lz))

N²ₛ = 1e-4
ϵ = 0.5
Dᴴ = ϵ * N²ₛ * Lz
Mᴴ = Dᴴ - N²ₛ * Lz

buoyancy = Oceananigans.BuoyancyFormulations.MoistConditionalBuoyancy(N²ₛ)
model = NonhydrostaticModel(; grid, buoyancy, tracers=(:D, :M), advection=WENO())

Dᵢ(x, y, z) = Dᴴ * z + 1e-2 * Dᴴ * Lz * randn()
Mᵢ(x, y, z) = Mᴴ * z + 1e-2 * Mᴴ * Lz * randn()
set!(model, D=Dᵢ, M=Mᵢ)

simulation = Simulation(model, Δt=1, stop_iteration=100)
conjure_time_step_wizard!(simulation, cfl=0.7)
run!(simulation)

