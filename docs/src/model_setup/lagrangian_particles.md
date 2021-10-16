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
grid = RegularRectilinearGrid(size=(10, 10, 10), extent=(1, 1, 1));

n_particles = 10;

x₀ = zeros(n_particles);

y₀ = rand(n_particles);

z₀ = -0.5 * ones(n_particles);

lagrangian_particles = LagrangianParticles(x=x₀, y=y₀, z=z₀)

# output
10 Lagrangian particles with
├── 3 properties: (:x, :y, :z)
└── 0 tracked fields: ()
```

then pass it to a model constructor

```jldoctest particles
model = NonhydrostaticModel(grid=grid, particles=lagrangian_particles, buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))

# output
NonhydrostaticModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=10, Ny=10, Nz=10)
├── tracers: (:T, :S)
├── closure: Nothing
├── buoyancy: SeawaterBuoyancy{Float64, LinearEquationOfState{Float64}, Nothing, Nothing}
├── coriolis: Nothing
└── particles: 10 Lagrangian particles with 3 properties: (:x, :y, :z)
```

!!! warn "Lagrangian particles on GPUs"
    Remember to use `CuArray` instead of regular `Array` when storing particle locations and properties on the GPU.

## Custom particles

If you want to keep track of custom properties, such as the species or DNA of a Lagrangian particle
representing a microbe in an agent-based model, then you can create your own custom particle type
and pass a `StructArray` to the `LagrangianParticles` constructor.

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

n_particles = 3;

x₀ = zeros(n_particles);

y₀ = rand(n_particles);

z₀ = -0.5 * ones(n_particles);

species = [:rock, :paper, :scissors]

dna = ["TATACCCC", "CCTAGGAC", "CGATTTAA"]

particles = StructArray{LagrangianMicrobe}((x₀, y₀, z₀, species, dna));

lagrangian_particles = LagrangianParticles(particles)

# output
3 Lagrangian particles with
├── 5 properties: (:x, :y, :z, :species, :dna)
└── 0 tracked fields: ()
```

!!! warn "Custom properties on GPUs"
    Not all data types can be passed to GPU kernels. If you intend to advect particles on the GPU make sure
    particle properties consist of only simple data types. The symbols and strings in this example won't
    work on the GPU.

## Writing particle properties to disk

Particle properties can be written to disk using JLD2 or NetCDF.

When writing to JLD2 you can pass `model.particles` as part of the named tuple of outputs.

```julia
JLD2OutputWriter(model, (particles=model.particles,), prefix="particles", schedule=TimeInterval(15))
```

When writing to NetCDF you should write particles to a separate file as the NetCDF dimensions differ for
particle trajectories. You can just pass `model.particles` straight to `NetCDFOutputWriter`:

```julia
NetCDFOutputWriter(model, model.particles, filepath="particles.nc", schedule=TimeInterval(15))
```

!!! warn "Outputting custom particle properties to NetCDF"
    NetCDF does not support arbitrary data types. If you need to write custom particle properties to disk
    that are not supported by NetCDF then you should use JLD2 (which should support almost any Julia data type).
