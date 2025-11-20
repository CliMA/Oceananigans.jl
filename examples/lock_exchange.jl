using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: MutableVerticalDiscretization


# Set resolution of the simulation grid 
Nx, Nz = 256, 64

# Set grid size 
L = 8kilometers   # horizontal length
H = 50meters      # depth  

# Allow for mutable surface height 
z_disc  = MutableVerticalDiscretization((-50, 0))

# Initialize the grid 
underlying_grid = RectilinearGrid(
                    size = (Nx, Nz),
                    x = (0, L),
                    z = z_disc,
                    topology = (Bounded, Flat, Bounded), 
                    halo = (5, 5)
)

# Add a slope at the bottom of the grid 
h_left = -H
h_right = -25meters
slope = (h_right - h_left) / L
bottom(x) = h_left + slope * x

# Use an immersed boundary with grid fitted bottom to describe the sloped bottom of the domain 
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

# Initialize the model 
    # Want to use a hydrostatic model 
    # Tracers act as markers within the fluid to track movement and dispersion
    # Weighted Essentially Non-Oscillatory (WENO) methods are useful for capturing sharp changes in density
    # Closures simplify complex systems by approximating the effects of unresolved scales
    # ZStarCoordinate allows for the top of the grid to move with the free surface
    # Runge Kutta good for integrating multiple processes 

model = HydrostaticFreeSurfaceModel(; grid,
    tracers = (:b,),
    buoyancy = BuoyancyTracer(),
    momentum_advection = WENO(order=5), 
    tracer_advection = WENO(order=7), 
    closure = (VerticalScalarDiffusivity(ν=1e-4), HorizontalScalarDiffusivity(ν=1.0)), 
    vertical_coordinate = ZStarCoordinate(grid), 
    free_surface = SplitExplicitFreeSurface(grid; substeps=10), 
    timestepper = :SplitRungeKutta3 
)

# Set initial conditions for lock exchange with different densities 
bᵢ(x, z) = x > 4kilometers ? 0.06 : 0.01
set!(model, b=bᵢ)

# Set the timesteps 
Δt = 5minutes
stop_time = 3days
simulation = Simulation(model; Δt, stop_time)

# Run simulation
run!(simulation)