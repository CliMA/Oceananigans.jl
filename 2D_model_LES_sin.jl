#using Pkg
using Oceananigans
using Printf
using Statistics

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using SeawaterPolynomials
using SeawaterPolynomials:TEOS10

using Glob

# Architecture
model_architecture = GPU()

# number of grid points
Ny = 20000
Nz = 256

const Ly = 40kilometers
const Lz = 512meters

grid = RectilinearGrid(model_architecture,
                       topology = (Flat, Bounded, Bounded),
                       size = (Ny, Nz),
                       halo = (5, 5),
                       y = (0, Ly),
                       z = (-Lz, 0))

@info "Built a grid: $grid."

#####
##### Boundary conditions
#####
const dTdz = 0.014
const dSdz = 0.0021

const T_surface = 20.0
const S_surface = 36.6
const max_temperature_flux = 2e-4

FILE_DIR = "./LES/NN_2D_channel_sin_cooling_$(max_temperature_flux)_LES"
mkpath(FILE_DIR)

@inline function temperature_flux(y, t)
    return max_temperature_flux * sin(π * y / Ly)
end

T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(temperature_flux))

#####
##### Coriolis
#####

const f₀ = 8e-5
coriolis = FPlane(f=f₀)

#####
##### Forcing and initial condition
#####
T_initial(y, z) = dTdz * z + T_surface
S_initial(y, z) = dSdz * z + S_surface

#####
##### Model building
#####

@info "Building a model..."

model = NonhydrostaticModel(; grid = grid,
                              advection = WENO(order=9),
                              coriolis = coriolis,
                              buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()),
                              tracers = (:T, :S),
                              timestepper = :RungeKutta3,
                              closure = nothing,
                              boundary_conditions = (; T=T_bcs))

@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
noise(z) = rand() * exp(z / 8)

T_initial_noisy(y, z) = T_initial(y, z) + 1e-6 * noise(z)
S_initial_noisy(y, z) = S_initial(y, z) + 1e-6 * noise(z)

set!(model, T=T_initial_noisy, S=S_initial_noisy)
#####
##### Simulation building
#####
simulation = Simulation(model, Δt = 0.1, stop_time = 30days)

# add timestep wizard callback
wizard = TimeStepWizard(cfl=0.6, max_change=1.05, max_Δt=20minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# add progress callback
wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): %6.3e, max(v): %6.3e, max(T): %6.3e, max(S): %6.3e, next Δt: %s\n",
        100 * (sim.model.clock.time / sim.stop_time),
        sim.model.clock.iteration,
        prettytime(sim.model.clock.time),
        prettytime(1e-9 * (time_ns() - wall_clock[1])),
        maximum(sim.model.velocities.u),
        maximum(sim.model.velocities.v),
        maximum(sim.model.tracers.T),
        maximum(sim.model.tracers.S),
        prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(1000))

#####
##### Diagnostics
#####

u, w = model.velocities.u, model.velocities.w
v = @at (Center, Center, Center) model.velocities.v
T, S = model.tracers.T, model.tracers.S

outputs = (; u, v, w, T, S)

#####
##### Build checkpointer and output writer
#####
simulation.output_writers[:jld2] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields.jld2",
                                                    schedule = TimeInterval(1hour))

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(1day),
                                                        prefix = "$(FILE_DIR)/checkpointer",
                                                        overwrite_existing = true)

@info "Running the simulation..."

try
    files = readdir(FILE_DIR)
    checkpoint_files = files[occursin.("checkpointer_iteration", files)]
    if !isempty(checkpoint_files)
        checkpoint_iters = parse.(Int, [filename[findfirst("iteration", filename)[end]+1:findfirst(".jld2", filename)[1]-1] for filename in checkpoint_files])
        pickup_iter = maximum(checkpoint_iters)
        run!(simulation, pickup="$(FILE_DIR)/checkpointer_iteration$(pickup_iter).jld2")
    else
        run!(simulation)
    end
catch err
    @info "run! threw an error! The error message is"
    showerror(stdout, err)
end

checkpointers = glob("$(FILE_DIR)/checkpointer_iteration*.jld2")
if !isempty(checkpointers)
    rm.(checkpointers)
end

# #####
# ##### Visualization
# #####
#%%
using CairoMakie


u_data = FieldTimeSeries("./NN_2D_channel_sin_cooling_$(max_temperature_flux)_LES.jld2", "u", backend=OnDisk())
v_data = FieldTimeSeries("./NN_2D_channel_sin_cooling_$(max_temperature_flux)_LES.jld2", "v", backend=OnDisk())
T_data = FieldTimeSeries("./NN_2D_channel_sin_cooling_$(max_temperature_flux)_LES.jld2", "T", backend=OnDisk())
S_data = FieldTimeSeries("./NN_2D_channel_sin_cooling_$(max_temperature_flux)_LES.jld2", "S", backend=OnDisk())

yC = ynodes(T_data.grid, Center())
yF = ynodes(T_data.grid, Face())

zC = znodes(T_data.grid, Center())
zF = znodes(T_data.grid, Face())

Nt = length(T_data.times)
#%%
fig = Figure(size = (1500, 900))
axu = CairoMakie.Axis(fig[1, 1], xlabel = "y (m)", ylabel = "z (m)", title = "u")
axv = CairoMakie.Axis(fig[1, 3], xlabel = "y (m)", ylabel = "z (m)", title = "v")
axT = CairoMakie.Axis(fig[2, 1], xlabel = "y (m)", ylabel = "z (m)", title = "Temperature")
axS = CairoMakie.Axis(fig[2, 3], xlabel = "y (m)", ylabel = "z (m)", title = "Salinity")
n = Obeservable(1)

uₙ = @lift interior(u_data[$n], 1, :, :)
vₙ = @lift interior(v_data[$n], 1, :, :)
Tₙ = @lift interior(T_data[$n], 1, :, :)
Sₙ = @lift interior(S_data[$n], 1, :, :)

ulim = @lift (-maximum([maximum(abs, $uₙ), 1e-16]), maximum([maximum(abs, $uₙ),  1e-16]))
vlim = @lift (-maximum([maximum(abs, $vₙ), 1e-16]), maximum([maximum(abs, $vₙ), 1e-16]))
Tlim = (minimum(interior(T_data[1])), maximum(interior(T_data[1])))
Slim = (minimum(interior(S_data[1])), maximum(interior(S_data[1])))

title_str = @lift "Time: $(round(T_data.times[$n] / 86400, digits=2)) days"
Label(fig[0, :], title_str, tellwidth = false)

hu = heatmap!(axu, yC, zC, uₙ, colormap=:RdBu_9, colorrange=ulim)
hv = heatmap!(axv, yC, zC, vₙ, colormap=:RdBu_9, colorrange=vlim)
hT = heatmap!(axT, yC, zC, Tₙ, colorrange=Tlim)
hS = heatmap!(axS, yC, zC, Sₙ, colorrange=Slim)

Colorbar(fig[1, 2], hu, label = "(m/s)")
Colorbar(fig[1, 4], hv, label = "(m/s)")
Colorbar(fig[2, 2], hT, label = "(°C)")
Colorbar(fig[2, 4], hS, label = "(psu)")

CairoMakie.record(fig, "$(FILE_DIR)/2D_sin_cooling_$(max_temperature_flux)_30days.mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end

# display(fig)
#%%