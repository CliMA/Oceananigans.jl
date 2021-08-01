using Oceananigans
using Oceananigans.Units
using Printf
using GLMakie
using Statistics

include("coupled_atmosphere_ocean_model.jl")

#####
##### Settings common to both atmos and ocean
#####

arch = CPU()
Nx = 128
Nz = 128

Lx = 1kilometer

Lz_atmos = 1kilometer
Lz_ocean = 100 # meters

coriolis = FPlane(latitude=45)

#####
##### Atmos model setup
#####

U₀ = 10 # initial and top velocity

atmos_grid = RegularRectilinearGrid(size = (Nx, Nz),
                                    x = (0, Lx),
                                    z = (0, Lz_atmos),
                                    topology = (Periodic, Flat, Bounded))

# Store boundary fluxes in arrays (ReducedField would be even better)
atmos_surface_flux_u = arch isa CPU ? zeros(atmos_grid.Nx, atmos_grid.Ny) : CUDA.zeros(atmos_grid.Nx, atmos_grid.Ny)
atmos_surface_flux_v = arch isa CPU ? zeros(atmos_grid.Nx, atmos_grid.Ny) : CUDA.zeros(atmos_grid.Nx, atmos_grid.Ny)

# Atmos velocity boundary conditions
atmos_u_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(atmos_surface_flux_u),
                                      top = ValueBoundaryCondition(U₀))

atmos_v_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(atmos_surface_flux_v))

atmos_equation_of_state = LinearEquationOfState(α = 2e-4)
     
atmos_model = NonhydrostaticModel(architecture = arch,
                                  grid = atmos_grid,
                                  timestepper = :RungeKutta3,
                                  advection = UpwindBiasedFifthOrder(),
                                  boundary_conditions = (u=atmos_u_bcs, v=atmos_v_bcs),
                                  closure = IsotropicDiffusivity(ν=1e-3, κ=1e-3),
                                  coriolis = coriolis,
                                  tracers = :T,
                                  buoyancy = SeawaterBuoyancy(constant_salinity = true,
                                                              equation_of_state = atmos_equation_of_state))

uᵢ(x, y, z) = U₀ * (1 + 1e-3 * randn())
set!(atmos_model, u=uᵢ)

#####
##### Ocean model setup
#####

ocean_grid = RegularRectilinearGrid(size = (Nx, Nz),
                                    x = (0, Lx),
                                    z = (-Lz_ocean, 0),
                                    topology = (Periodic, Flat, Bounded))

# Make sure horizontal grids are the same
@assert ocean_grid.Nx == atmos_grid.Nx
@assert ocean_grid.Ny == atmos_grid.Ny
@assert ocean_grid.Lx == atmos_grid.Lx
@assert ocean_grid.Ly == atmos_grid.Ly

# Ocean velocity boundary conditions
ocean_surface_flux_u = arch isa CPU ? zeros(atmos_grid.Nx, atmos_grid.Ny) : CUDA.zeros(atmos_grid.Nx, atmos_grid.Ny)
ocean_surface_flux_v = arch isa CPU ? zeros(atmos_grid.Nx, atmos_grid.Ny) : CUDA.zeros(atmos_grid.Nx, atmos_grid.Ny)

ocean_u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(ocean_surface_flux_u))
ocean_v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(ocean_surface_flux_v))

ocean_equation_of_state = LinearEquationOfState(α=2e-4)
    
ocean_model = NonhydrostaticModel(architecture = arch,
                                  grid = ocean_grid,
                                  timestepper = :RungeKutta3,
                                  advection = UpwindBiasedFifthOrder(),
                                  boundary_conditions = (u=ocean_u_bcs, v=ocean_v_bcs),
                                  closure = IsotropicDiffusivity(ν=1e-3, κ=1e-3),
                                  coriolis = coriolis,
                                  tracers = (:T, :S),
                                  buoyancy = SeawaterBuoyancy(equation_of_state = ocean_equation_of_state))

#####
##### Coupled model + simulation!
#####

coupled_model = CoupledAtmosphereOceanModel(atmos_model, ocean_model)

simulation = Simulation(coupled_model, Δt=1.0, stop_time=12hours)

# Progress logging callback

function print_progress(sim)
    uo, vo, wo = sim.model.ocean.velocities
    ua, va, wa = sim.model.atmos.velocities
    iter = sim.model.clock.iteration
    time = sim.model.clock.time

    atmos_surface_flux_u = ua.boundary_conditions.bottom.condition
    ocean_surface_flux_u = uo.boundary_conditions.top.condition

    msg1 = @sprintf("Iter: % 5d, time: % 14s, next Δt: % 14s, <τₓ> (atmos, ocean): (%.2e, %.2e), ",
                    iter,
                    prettytime(time),
                    prettytime(sim.Δt),
                    mean(atmos_surface_flux_u),
                    mean(ocean_surface_flux_u))

    msg2 = @sprintf("uo: (%.2e, %.2e), wo: (%.2e, %.2e), ua: (%.2e, %.2e), wa: (%.2e, %.2e)",
                    minimum(uo), maximum(uo),
                    minimum(wo), maximum(wo),
                    minimum(ua), maximum(ua),
                    minimum(wa), maximum(wa))

    @info msg1 * msg2

    return nothing
end

# Adaptive time-stepping callback

using Oceananigans.Simulations: update_Δt!

wizard = TimeStepWizard(cfl=0.7, Δt=1second) 

function update_simulation_Δt!(sim)
    update_Δt!(wizard, sim.model.atmos) # use atmos to set Δt
    sim.Δt = wizard.Δt
    return nothing
end

simulation.callbacks[:progress] = Callback(print_progress, schedule=IterationInterval(100))
simulation.callbacks[:wizard] = Callback(update_simulation_Δt!, schedule=IterationInterval(10))

#####
##### Output
#####

simulation.output_writers[:atmos] = JLD2OutputWriter(atmos_model,
                                                     merge(atmos_model.velocities, atmos_model.tracers),
                                                     schedule = TimeInterval(10minutes),
                                                     prefix = "coupled_model_atmos",
                                                     force = true,
                                                     field_slicer = nothing)

simulation.output_writers[:ocean] = JLD2OutputWriter(ocean_model,
                                                     merge(ocean_model.velocities, ocean_model.tracers),
                                                     schedule = TimeInterval(10minutes),
                                                     prefix = "coupled_model_ocean",
                                                     force = true,
                                                     field_slicer = nothing)

run!(simulation)

######
###### Visualize
######

u_ocean = FieldTimeSeries("coupled_model_ocean.jld2", "u")
u_atmos = FieldTimeSeries("coupled_model_atmos.jld2", "u")
w_ocean = FieldTimeSeries("coupled_model_ocean.jld2", "w")
w_atmos = FieldTimeSeries("coupled_model_atmos.jld2", "w")

fig = Figure(backgroundcolor = RGBf0(0.98, 0.98, 0.98), resolution = (2000, 1000))
u_axs = fig[1:2, 1] = [Axis(fig, title = t) for t in ["Atmos u", "Ocean u"]]
w_axs = fig[1:2, 2] = [Axis(fig, title = t) for t in ["Atmos w", "Ocean w"]]
ax_u_atmos, ax_u_ocean = u_axs
ax_w_atmos, ax_w_ocean = w_axs

n = Node(1) 

ua = @lift interior(u_atmos[$n])[:, 1, :]
uo = @lift interior(u_ocean[$n])[:, 1, :]

wa = @lift interior(w_atmos[$n])[:, 1, :]
wo = @lift interior(w_ocean[$n])[:, 1, :]

heatmap!(ax_u_atmos, ua)
heatmap!(ax_u_ocean, uo)

heatmap!(ax_w_atmos, wa)
heatmap!(ax_w_ocean, wo)

record(fig, "coupled_model.mp4", 1:length(u_ocean.times); framerate = 8) do save_point
    n[] = save_point
end

# display(fig)

