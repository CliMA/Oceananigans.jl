using Oceananigans
using Oceananigans.Units

Nz = 32; β = 10; N² = 4e-5
grid = RectilinearGrid(CPU(), size=(Nz, Nz, Nz), extent = (10Nz, 10Nz, 2Nz))
model = @show NonhydrostaticModel(; grid, buoyancy = BuoyancyTracer(), tracers = :b)

@inline bᵢ(x, y, z) = z > -Nz ? N² * z : N² * ((z + Nz)/β - Nz)
set!(model, b = bᵢ)
b = model.tracers.b

using CairoMakie
scatter(Field(Average(b, dims=(1, 2))))
