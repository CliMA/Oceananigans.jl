# Lagrangian particle tracking

Models can keep track of the location and properties of neutrally buoyant particles. Particles are
advected with the flow field using forward Euler time-stepping at every model iteration.

## Simple particles

If you just need to keep of particle locations ``(x, y, z)`` then you can construct some Lagrangian particles
using the regular `LagrangianParticles` constructor

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```jldoctest particles
grid = RectilinearGrid(size=(10, 10, 10), extent=(1, 1, 1))

Nparticles = 10

x₀ = zeros(Nparticles)

y₀ = rand(Nparticles)

z₀ = -0.5 * ones(Nparticles)

lagrangian_particles = LagrangianParticles(x=x₀, y=y₀, z=z₀)

# output
10 LagrangianParticles with eltype Particle:
├── 3 properties: (:x, :y, :z)
├── boundary conditions: ParticleBoundaryConditions(ReflectingBoundary(1.0))
├── 0 tracked fields: ()
├── sorting: nothing
└── dynamics: no_dynamics
```

then pass it to a model constructor

```jldoctest particles
model = NonhydrostaticModel(grid; particles=lagrangian_particles)

# output
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 10×10×10 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── timestepper: RungeKutta3TimeStepper
├── advection scheme:
│   └── momentum: Centered(order=2)
├── tracers: ()
├── closure: Nothing
├── buoyancy: Nothing
├── coriolis: Nothing
└── particles: 10 LagrangianParticles with eltype Particle and properties (:x, :y, :z)
```

!!! warn "Lagrangian particles on GPUs"
    Remember to use `CuArray` instead of regular `Array` when storing particle locations and properties on the GPU.

## Custom particles

If you want to keep track of custom properties, such as the species or DNA of a Lagrangian particle
representing a microbe in an agent-based model, then you can create your own custom particle type
and pass a `StructArray` to the `LagrangianParticles` constructor.

Fields passed through `tracked_fields` are sampled before custom `dynamics` and refreshed after
advection and boundary handling. Recorded properties therefore match the recorded particle positions.

```jldoctest particles
using Oceananigans
using StructArrays

struct LagrangianMicrobe{T, S, D}
    x :: T
    y :: T
    z :: T
    species :: S
    dna :: D
end

Nparticles = 3

x₀ = zeros(Nparticles)

y₀ = rand(Nparticles)

z₀ = -0.5 * ones(Nparticles)

species = [:rock, :paper, :scissors]

dna = ["TATACCCC", "CCTAGGAC", "CGATTTAA"]

particles = StructArray{LagrangianMicrobe}((x₀, y₀, z₀, species, dna));

lagrangian_particles = LagrangianParticles(particles)

# output
3 LagrangianParticles with eltype LagrangianMicrobe:
├── 5 properties: (:x, :y, :z, :species, :dna)
├── boundary conditions: ParticleBoundaryConditions(ReflectingBoundary(1.0))
├── 0 tracked fields: ()
├── sorting: nothing
└── dynamics: no_dynamics
```

!!! warn "Custom properties on GPUs"
    Not all data types can be passed to GPU kernels. If you intend to advect particles on the GPU make sure
    particle properties consist of only simple data types. The symbols and strings in this example won't
    work on the GPU.

## Boundary conditions

By default, particles reflect off the boundaries of `Bounded` directions and off immersed boundaries.
Use `ParticleBoundaryConditions` to specify the behavior of particles at each boundary.
`ReflectingBoundary(restitution)` reflects particles with a coefficient of restitution, and
`AbsorbingBoundary()` removes particles that cross the boundary:

```jldoctest particles
boundary_conditions = ParticleBoundaryConditions(ReflectingBoundary(0.5); bottom=AbsorbingBoundary())
lagrangian_particles = LagrangianParticles(x=x₀, y=y₀, z=z₀; boundary_conditions)

# output
3 LagrangianParticles with eltype Particle:
├── 3 properties: (:x, :y, :z)
├── boundary conditions: ParticleBoundaryConditions(ReflectingBoundary(0.5); bottom=AbsorbingBoundary())
├── 0 tracked fields: ()
├── sorting: nothing
└── dynamics: no_dynamics
```

Absorbed particles are removed after they are advected, so the number of particles may decrease
as the simulation runs. The remaining particles keep their order.

## Sorting particles by grid cell

Particle interactions (for example, collisions) and particle-to-grid operations are efficient
when particles are stored in order of the grid cell they occupy. Pass a `sorting` algorithm to
`LagrangianParticles` to sort particles by cell at the beginning of every particle step:

```jldoctest particles
lagrangian_particles = LagrangianParticles(x=x₀, y=y₀, z=z₀, sorting=CountingSort())

# output
3 LagrangianParticles with eltype Particle:
├── 3 properties: (:x, :y, :z)
├── boundary conditions: ParticleBoundaryConditions(ReflectingBoundary(1.0))
├── 0 tracked fields: ()
├── sorting: CellSorting(CountingSort())
└── dynamics: no_dynamics
```

Two algorithms are available, and both produce the same order:

- `CountingSort()` counts and scatters particles into cells with atomic operations, and its cost
  is linear in the number of particles and cells.
- `ComparisonSort()` uses the general-purpose `sort!` for the particle array type.

After sorting, the particles in cell `(i, j, k)` are indexed by
`cell_particle_indices(i, j, k, grid, lagrangian_particles.sorting)`, which can be used in custom
`dynamics`. Within each cell, particles keep the order they had before sorting.

!!! warn "Sorting changes particle indices"
    Sorting reorders particle properties, so a particle's index may change from one time step
    to the next. To follow individual particles, add an identifying property to a custom particle type.

## Writing particle properties to disk

Particle properties can be written to disk using JLD2 or NetCDF.

When writing to JLD2 you can pass `model.particles` as part of the named tuple of outputs.

```@setup particles
using Oceananigans
using NCDatasets
grid = RectilinearGrid(size=(10, 10, 10), extent=(1, 1, 1))
Nparticles = 3
x₀ = zeros(Nparticles)
y₀ = rand(Nparticles)
z₀ = -0.5 * ones(Nparticles)
lagrangian_particles = LagrangianParticles(x=x₀, y=y₀, z=z₀)
model = NonhydrostaticModel(grid; particles=lagrangian_particles)
```

```@example particles
JLD2Writer(model, (; model.particles), filename="particles", schedule=TimeInterval(15))
```

When writing to NetCDF you should write particles to a separate file as the NetCDF dimensions differ for
particle trajectories. You can just pass `model.particles` straight to `NetCDFWriter`:

```@example particles
NetCDFWriter(model, (; model.particles), filename="particles.nc", schedule=TimeInterval(15))
```

!!! warn "Outputting custom particle properties to NetCDF"
    NetCDF does not support arbitrary data types. If you need to write custom particle properties to disk
    that are not supported by NetCDF then you should use JLD2 (which should support almost any Julia data type).
