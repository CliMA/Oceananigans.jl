using Oceananigans
using Adapt
using Oceananigans.Grids: AbstractUnderlyingGrid, R_Earth, generate_coordinate
using Oceananigans.Architectures: AbstractArchitecture, arch_array
using Oceananigans.Operators: hack_cosd, hack_sind

struct ToroidalGrid{FT, TZ, VX, VY, VZ, FZ, Arch} <: AbstractUnderlyingGrid{FT, Periodic, Periodic, TZ, Arch}
    architecture :: Arch
    Nx :: Int
    Ny :: Int
    Nz :: Int
    Hx :: Int
    Hy :: Int
    Hz :: Int
    Lx :: FT
    Ly :: FT
    Lz :: FT
    # All directions exept z ARE! regular (FX, FY, FZ) <: Number
    λᶠᵃᵃ  :: VX
    λᶜᵃᵃ  :: VX
    φᵃᶠᵃ  :: VY
    φᵃᶜᵃ  :: VY
    zᵃᵃᶠ  :: VZ
    zᵃᵃᶜ  :: VZ
    Δλ    :: FT
    Δφ    :: FT
    Δzᵃᵃᶠ :: FZ
    Δzᵃᵃᶜ :: FZ
    # Spherical radius
    inner_radius :: FT
    outer_radius :: FT

    ToroidalGrid{TZ}(arch::Arch,
                     Nx, Ny, Nz,
                     Hx, Hy, Hz,
                     Lx :: FT, Ly :: FT, Lz :: FT,
                      λᶠᵃᵃ :: VX,  λᶜᵃᵃ :: VX,
                      φᵃᶠᵃ :: VY,  φᵃᶜᵃ :: VY,
                      zᵃᵃᶠ :: VZ,  zᵃᵃᶜ :: VZ,
                      Δλ   :: FT,  Δφ   :: FT,
                      Δzᵃᵃᶠ:: FZ, Δzᵃᵃᶜ :: FZ,
                     inner_radius :: FT,
                     outer_radius :: FT) where {Arch, FT,
                                                TZ, VX, VY,
                                                VZ, FZ} =
        new{FT, TZ, VX, VY, VZ, FZ, Arch}(arch, Nx, Ny, Nz,
                                          Hx, Hy, Hz, Lx, Ly, Lz,
                                          λᶠᵃᵃ,  λᶜᵃᵃ, φᵃᶠᵃ,  φᵃᶜᵃ,
                                          zᵃᵃᶠ,  zᵃᵃᶜ, Δλ  ,  Δφ, 
                                          Δzᵃᵃᶠ, Δzᵃᵃᶜ,   
                                          inner_radius,
                                          outer_radius)
end

function ToroidalGrid(arch::AbstractArchitecture = CPU(),
                      FT::DataType = Float64;
                      size,
                      z = (-1, 0),
                      halo = (3, 3, 3),
                      inner_radius = R_Earth,
                      outer_radius = R_Earth,
                      topology_z = Bounded)

    Nλ, Nφ, Nz = size
    Hλ, Hφ, Hz = halo

    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, Periodic(),   Nλ, Hλ, (-180, 180), arch)
    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, Periodic(),   Nφ, Hφ, (-180, 180), arch)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, topology_z(), Nz, Hz, z,           arch)

    Δλ = Δλᶠᵃᵃ
    Δφ = Δφᵃᶠᵃ

    return ToroidalGrid{topology_z}(arch, Nλ, Nφ, Nz,
                                    Hλ, Hφ, Hz, Lλ, Lφ, Lz,
                                    λᶠᵃᵃ,  λᶜᵃᵃ, φᵃᶠᵃ,  φᵃᶜᵃ,
                                    zᵃᵃᶠ,  zᵃᵃᶜ, Δλ  ,  Δφ,    
                                    Δzᵃᵃᶠ, Δzᵃᵃᶜ,
                                    FT(inner_radius),
                                    FT(outer_radius))
end

Adapt.adapt_structure(to, grid::ToroidalGrid{FT, TZ}) where {FT, TZ} = 
        ToroidalGrid{TZ}(nothing,
                         grid.Nx, grid.Ny, grid.Nz,
                         grid.Hx, grid.Hy, grid.Hz,
                         grid.Lx, grid.Ly, grid.Lz,
                         Adapt.adapt(to, grid.λᶠᵃᵃ), Adapt.adapt(to, grid.λᶜᵃᵃ),
                         Adapt.adapt(to, grid.φᵃᶠᵃ), Adapt.adapt(to, grid.φᵃᶜᵃ),
                         Adapt.adapt(to, grid.zᵃᵃᶠ), Adapt.adapt(to, grid.zᵃᵃᶜ),
                         grid.Δλ   ,  grid.Δφ  ,
                         Adapt.adapt(to, grid.Δzᵃᵃᶠ), Adapt.adapt(to, grid.Δzᵃᵃᶜ),
                         grid.inner_radius,
                         grid.outer_radius)


import Oceananigans.Grids: on_architecture

function on_architecture(new_arch, old_grid::ToroidalGrid{FT, TZ}) where {FT, TZ}
    old_properties = (old_grid.λᶠᵃᵃ,  old_grid.λᶜᵃᵃ,
                      old_grid.φᵃᶠᵃ,  old_grid.φᵃᶜᵃ,
                      old_grid.zᵃᵃᶠ,  old_grid.zᵃᵃᶜ)

    new_properties = Tuple(arch_array(new_arch, p) for p in old_properties)

    return ToroidalGrid{TZ}(new_arch,
                            old_grid.Nx, old_grid.Ny, old_grid.Nz,
                            old_grid.Hx, old_grid.Hy, old_grid.Hz,
                            old_grid.Lx, old_grid.Ly, old_grid.Lz,
                            new_properties...,
                            old_grid.Δλ, old_grid.Δφ, 
                            arch_array(new_arch, grid.Δzᵃᵃᶠ), 
                            arch_array(new_arch, grid.Δzᵃᵃᶜ), 
                            old_grid.inner_radius,
                            old_grid.outer_radius)
end

import Oceananigans.Operators: Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δxᶜᶜᵃ
import Oceananigans.Operators: Δyᵃᶠᵃ, Δyᵃᶜᵃ
import Oceananigans.Operators: Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ
import Oceananigans.Operators: Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ

## On the fly metrics
@inline Δxᶠᶜᵃ(i, j, k, grid::ToroidalGrid) = @inbounds deg2rad(grid.Δλ) * (grid.outer_radius - grid.inner_radius * hack_cosd(grid.φᵃᶜᵃ[j]))
@inline Δxᶜᶠᵃ(i, j, k, grid::ToroidalGrid) = @inbounds deg2rad(grid.Δλ) * (grid.outer_radius - grid.inner_radius * hack_cosd(grid.φᵃᶠᵃ[j]))
@inline Δxᶠᶠᵃ(i, j, k, grid::ToroidalGrid) = @inbounds deg2rad(grid.Δλ) * (grid.outer_radius - grid.inner_radius * hack_cosd(grid.φᵃᶠᵃ[j]))
@inline Δxᶜᶜᵃ(i, j, k, grid::ToroidalGrid) = @inbounds deg2rad(grid.Δλ) * (grid.outer_radius - grid.inner_radius * hack_cosd(grid.φᵃᶜᵃ[j]))

@inline Δyᵃᶠᵃ(i, j, k, grid::ToroidalGrid) = @inbounds grid.inner_radius * deg2rad(grid.Δφ)
@inline Δyᵃᶜᵃ(i, j, k, grid::ToroidalGrid) = @inbounds grid.inner_radius * deg2rad(grid.Δφ)

@inline Azᶠᶜᵃ(i, j, k, grid::ToroidalGrid) = @inbounds grid.inner_radius * grid.outer_radius * deg2rad(grid.Δλ) * deg2rad(grid.Δφ)
@inline Azᶜᶠᵃ(i, j, k, grid::ToroidalGrid) = @inbounds grid.inner_radius * grid.outer_radius * deg2rad(grid.Δλ) * deg2rad(grid.Δφ)
@inline Azᶠᶠᵃ(i, j, k, grid::ToroidalGrid) = @inbounds grid.inner_radius * grid.outer_radius * deg2rad(grid.Δλ) * deg2rad(grid.Δφ)
@inline Azᶜᶜᵃ(i, j, k, grid::ToroidalGrid) = @inbounds grid.inner_radius * grid.outer_radius * deg2rad(grid.Δλ) * deg2rad(grid.Δφ)

import Oceananigans.Models.HydrostaticFreeSurfaceModels: validate_momentum_advection

validate_momentum_advection(advection, grid::ToroidalGrid) = advection

import Oceananigans.Grids: node, λnode, φnode, znode

@inline node(i, j, k, grid::ToroidalGrid, ℓx, ℓy, ℓz) = (λnode(i, j, k, grid, ℓx, ℓy, ℓz),
                                                       φnode(i, j, k, grid, ℓx, ℓy, ℓz),
                                                       znode(i, j, k, grid, ℓx, ℓy, ℓz))

@inline node(i, j, k, grid::ToroidalGrid, ℓx, ℓy, ℓz::Nothing) = (λnode(i, j, k, grid, ℓx, ℓy, ℓz), φnode(i, j, k, grid, ℓx, ℓy, ℓz))

@inline λnode(i, grid::ToroidalGrid, ::Center) = @inbounds grid.λᶜᵃᵃ[i]
@inline λnode(i, grid::ToroidalGrid, ::Face)   = @inbounds grid.λᶠᵃᵃ[i]
@inline φnode(j, grid::ToroidalGrid, ::Center) = @inbounds grid.φᵃᶜᵃ[j]
@inline φnode(j, grid::ToroidalGrid, ::Face)   = @inbounds grid.φᵃᶠᵃ[j]
@inline znode(k, grid::ToroidalGrid, ::Center) = @inbounds grid.zᵃᵃᶜ[k]
@inline znode(k, grid::ToroidalGrid, ::Face)   = @inbounds grid.zᵃᵃᶠ[k]

# convenience
@inline λnode(i, j, k, grid::ToroidalGrid, ℓx, ℓy, ℓz) = λnode(i, grid, ℓx)
@inline φnode(i, j, k, grid::ToroidalGrid, ℓx, ℓy, ℓz) = φnode(j, grid, ℓy)
@inline znode(i, j, k, grid::ToroidalGrid, ℓx, ℓy, ℓz) = znode(k, grid, ℓz)

function toroidal_to_cartesian(λ, φ, z, grid)
    x = hack_cosd(λ) * (grid.outer_radius - grid.inner_radius * hack_cosd(φ))
    y = hack_sind(λ) * (grid.outer_radius - grid.inner_radius * hack_cosd(φ))
    z = grid.inner_radius * hack_sind(φ)

    return x, y, z
end

function toroidal2cartesian(λ, φ, rᵢ=1, rₒ=10)
    Nλ = length(λ)
    Nφ = length(φ)

    λa = repeat(reshape(λ, Nλ, 1), 1, Nφ) 
    φa = repeat(reshape(φ, 1, Nφ), Nλ, 1)

    x = @. hack_cosd(λa) * (rₒ - rᵢ * hack_cosd(φa))
    y = @. hack_sind(λa) * (rₒ - rᵢ * hack_cosd(φa))
    z = @. hack_sind(φa) * rᵢ

    return x, y, z
end

### Let's run the simulation!

grid = ToroidalGrid(GPU(), size = (2048, 512, 1), inner_radius = 3, outer_radius = 7, halo = (7, 7, 7))

model = HydrostaticFreeSurfaceModel(; grid,
                                      momentum_advection = VectorInvariant(; vorticity_scheme = WENO(order = 9)),
                                      free_surface = SplitExplicitFreeSurface(; grid, cfl = 0.7),
                                      buoyancy = nothing,
                                      closure = nothing,
                                      coriolis = nothing,
                                      tracers = ())

set!(model, u = (x, y, z) -> rand(), v = (x, y, z) -> rand())

u, v, w = model.velocities
interior(u) .-= mean(u)
interior(v) .-= mean(v)

simulation = Simulation(model, Δt = 1e-3, stop_time = 100)

wtime = Ref(time_ns())

function progress(sim)
    @info @sprintf("iteration: %d, wall time: %s, (|u|, |v|, |w|): %.2e, %.2e, %.2e \n", sim.model.clock.iteration, prettytime((time_ns() - wtime[])*1e-9), maximum(abs, sim.model.velocities.u), maximum(abs, sim.model.velocities.v), maximum(abs, sim.model.velocities.w))
    wtime[] = time_ns()
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

using Oceananigans.Operators: ζ₃ᶠᶠᶜ

ζ = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, u, v)

simulation.output_writers[:surface] = JLD2OutputWriter(model, (; u, v, w, ζ);
                                                       filename = "torus",
                                                       schedule = TimeInterval(0.1))

run!(simulation)
