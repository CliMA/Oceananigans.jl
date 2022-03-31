using Random
using Printf
using Oceananigans
using Oceananigans.Units: minute, minutes, hour

# Stretched grid
Lz = 32          # (m) domain depth
Lx = Ly = 64     # domain width    
Nx = Ny = 32
Nz = 24          # number of points in the vertical direction

refinement = 1.2 # controls spacing near surface (higher means finer spaced)
stretching = 12  # controls rate of stretching at bottom
h(k) = (k - 1) / Nz
ζ₀(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

# Vertically-stretched and uniform options
z_stretched(k) = Lz * (ζ₀(k) * Σ(k) - 1)
z_uniform = (-Lz, 0)

grid = RectilinearGrid(; size = (Nx, Ny, Nz), halo=(3, 3, 3),
                       x = (-Lx/2, Lx/2),
                       y = (-Ly/2, Lx/2),
                       z = z_stretched)

@info "Build a grid:"
@show grid

# 10 Lagrangian particles
Nparticles = 10
x₀ = Lx / 10 * (2rand(Nparticles) .- 1)
y₀ = Ly / 10 * (2rand(Nparticles) .- 1)
z₀ = - Lz / 10 * rand(Nparticles)
particles = LagrangianParticles(x=x₀, y=y₀, z=z₀, restitution=0)

@info "Initialized Lagrangian particles"
@show particles

# Convection
b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(1e-8))

model = NonhydrostaticModel(; grid, particles,
                            advection = UpwindBiasedFifthOrder(),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = AnisotropicMinimumDissipation(),
                            boundary_conditions = (; b=b_bcs))

@info "Constructed a model"
@show model

bᵢ(x, y, z) = 1e-5 * z + 1e-9 * rand()
set!(model, b=bᵢ)

simulation = Simulation(model, Δt=10.0, stop_time=20minutes)
wizard = TimeStepWizard(cfl=0.5, max_change=1.1, max_Δt=1minute)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

run!(simulation)

