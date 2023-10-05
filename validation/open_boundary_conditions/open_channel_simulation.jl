using Oceananigans
using Oceananigans.Operators
using Oceananigans.OutputReaders
using Oceananigans.OutputReaders: OnDisk
using Oceananigans.Units
using Printf

include("generate_input_data.jl")

#####
##### Baroclinic double gyre simulation with one-way-coupled air-sea interaction
#####

# Simulation parameters

stop_time = 365days
forcing_frequency = 1day

arch = CPU()

# Defining the grid 
grid = LatitudeLongitudeGrid(arch;
                             size = (100, 100, 10), 
                         latitude = (15, 75), 
                        longitude = (0, 60), 
                             halo = (4, 4, 4),
                                z = (-1000, 0))

#####
##### Generate Input Data
#####

# Create a time series of atmospheric data with timestep 1 day
# running for 1 year.
times = range(0, stop_time, step = forcing_frequency)
boundary_file = "boundary_data.jld2"

generate_input_data!(grid, times, boundary_file)

#####
##### Define Boundary Conditions
#####

T_top = FieldTimeSeries(boundary_file, "T_top"; backend = InMemory(; chunk_size = 10))
u_top = FieldTimeSeries(boundary_file, "u_top"; backend = InMemory(; chunk_size = 10))

T_west = FieldTimeSeries(boundary_file, "T_west"; backend = InMemory(; chunk_size = 10))
u_west = FieldTimeSeries(boundary_file, "u_west"; backend = InMemory(; chunk_size = 10))

T_top_bc = ValueBoundaryCondition(T_top)
u_top_bc = ValueBoundaryCondition(u_top)

T_west_bc = OpenBoundaryCondition(T_west)
u_west_bc = OpenBoundaryCondition(u_west)

T_bcs = FieldBoundaryConditions(top = T_top_bc, west = T_west_bc)
u_bcs = FieldBoundaryConditions(top = u_top_bc, west = u_west_bc)

#####
##### Physical and Numerical Setup
#####

momentum_advection = VectorInvariant(vorticity_scheme = WENO(), 
                                      vertical_scheme = WENO())

tracer_advection = WENO()

buoyancy = SeawaterBuoyancy(equation_of_state = LinearEquationOfState(), constant_salinity = 35)

free_surface = SplitExplicitFreeSurface(; grid, cfl = 0.7, substeps = nothing)

coriolis = HydrostaticSphericalCoriolis()

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 0.1)
vertical_diffusivity  = VerticalScalarDiffusivity(κ = 1e-5, ν = 1e-4)

closure = (convective_adjustment, vertical_diffusivity)

#####
##### Create the model and initial conditions
#####

model = HydrostaticFreeSurfaceModel(; 
                                     grid, 
                                     momentum_advection,
                                     tracer_advection,
                                     tracers = :T,
                                     free_surface, buoyancy, coriolis, closure,
                                     boundary_conditions = (T = T_bcs, u = u_bcs))

Tᵢ(x, y, z) = 2 * (1 + z / 1000)
                                    
set!(model, T = Tᵢ)

#####
##### Simulation setup
#####

simulation = Simulation(model; Δt = 15minutes, stop_time)

function progress(sim)
    model = sim.model
    u, v, w = model.velocities
    T = model.tracers.T 
    @info @sprintf("Simulation time: %s, max(|u|, |v|, |w|, |T|): %.2e, %.2e, %.2e, %.2e \n", 
                   prettytime(sim.model.clock.time), 
                   maximum(abs, u), maximum(abs, v), 
                   maximum(abs, w), maximum(abs, T))

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers);
                                                      schedule = TimeInterval(1day), 
                                                      filename = "double_gyre",
                                                      overwrite_existing = true)