using Oceananigans
using Oceananigans.MultiRegion
using Oceananigans.MultiRegion: getregion, propagate_region, getdevice, switch_device!
using Oceananigans.BoundaryConditions
using Oceananigans.Fields: set!
using BenchmarkTools

using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface
using Statistics
using Printf
using LinearAlgebra, SparseArrays
using Oceananigans.Solvers: constructors, unpack_constructors

Lh = 100kilometers
Lz = 400meters
Nx = 64

grid = RectilinearGrid(GPU(),
    size = (Nx, 3, 1),
    x = (0, Lh), y = (0, Lh), z = (-Lz, 0),
    topology = (Periodic, Periodic, Bounded))

mrg = MultiRegionGrid(grid, partition = XPartition(2), devices = (0, 2))

coriolis = FPlane(f = 1e-4)

model = HydrostaticFreeSurfaceModel(grid = mrg,
    coriolis = coriolis)

gaussian(x, L) = exp(-x^2 / 2L^2)

U = 0.1 # geostrophic velocity
L = Lh / 40 # gaussian width
x₀ = Lh / 4 # gaussian center

vᵍ(x, y, z) = -U * (x - x₀) / L * gaussian(x - x₀, L)

g = model.free_surface.gravitational_acceleration

η₀ = coriolis.f * U * L / g # geostrohpic free surface amplitude

ηᵍ(x) = η₀ * gaussian(x - x₀, L)

ηⁱ(x, y) = 2 * ηᵍ(x)

set!(model, v = vᵍ)
set!(model, η = ηⁱ)

gravity_wave_speed = sqrt(g * Lz) # hydrostatic (shallow water) gravity wave speed
Δt = model.grid.region_grids[2].Δxᶜᵃᵃ / gravity_wave_speed

timesteps = 20

vres = zeros(Nx, timesteps)

Nh = Int(Nx/2)
for i in 1:timesteps
    time_step!(model, Δt)
    switch_device!(getdevice(model, 1))
    vres[1:Nh, i] = Array(model.velocities.v.data[1].parent)[2:Nh+1, 1, 1]
    switch_device!(getdevice(model, 2))
    vres[Nh+1:Nx, i] = Array(model.velocities.v.data[2].parent)[2:Nh+1, 1, 1]
    @show i
end
