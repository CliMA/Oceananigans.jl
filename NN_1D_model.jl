#using Pkg
# pkg"add Oceananigans CairoMakie"
using Oceananigans
include("NN_closure_global.jl")
include("xin_kai_vertical_diffusivity_local.jl")
include("feature_scaling.jl")

ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Statistics

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using SeawaterPolynomials
using SeawaterPolynomials:TEOS10


# Architecture
model_architecture = CPU()

# number of grid points
Nz = 64
const Lz = 512

grid = RectilinearGrid(model_architecture,
    topology = (Flat, Flat, Bounded),
    size = Nz,
    halo = 3,
    z = (-Lz, 0))

@info "Built a grid: $grid."

#####
##### Boundary conditions
#####
const dTdz = 0.014
const dSdz = 0.0021

const T_surface = 20.0
const S_surface = 36.6

T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(3e-4))
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(2e-4))

#####
##### Coriolis
#####

const f₀ = 1e-4
const β = 1e-11
# coriolis = BetaPlane(f₀=f₀, β = β)
coriolis = FPlane(f=f₀)

#####
##### Forcing and initial condition
#####
T_initial(z) = dTdz * z + T_surface
S_initial(z) = dSdz * z + S_surface

nn_closure = NNFluxClosure(model_architecture)
base_closure = XinKaiLocalVerticalDiffusivity()

#####
##### Model building
#####

@info "Building a model..."

model = HydrostaticFreeSurfaceModel(
    grid = grid,
    free_surface = ImplicitFreeSurface(),
    momentum_advection = WENO(grid = grid),
    tracer_advection = WENO(grid = grid),
    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()),
    coriolis = coriolis,
    closure = (nn_closure, base_closure),
    # closure = base_closure,
    tracers = (:T, :S),
    # boundary_conditions = (; T = T_bcs),
    boundary_conditions = (; T = T_bcs, u = u_bcs),
)

@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
noise(z) = rand() * exp(z / 8)

T_initial_noisy(z) = T_initial(z) + 1e-6 * noise(z)
S_initial_noisy(z) = S_initial(z) + 1e-6 * noise(z)

set!(model, T=T_initial_noisy, S=S_initial_noisy)
using Oceananigans.TimeSteppers: update_state!
update_state!(model)
#####
##### Simulation building
#####
Δt₀ = 5minutes
stop_time = 10days

simulation = Simulation(model, Δt = Δt₀, stop_time = stop_time)

# add timestep wizard callback
# wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=20minutes)
# simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

# add progress callback
wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(T): %6.3e, max(S): %6.3e, next Δt: %s\n",
        100 * (sim.model.clock.time / sim.stop_time),
        sim.model.clock.iteration,
        prettytime(sim.model.clock.time),
        prettytime(1e-9 * (time_ns() - wall_clock[1])),
        maximum(abs, sim.model.tracers.T),
        maximum(abs, sim.model.tracers.S),
        prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(20))

#####
##### Diagnostics
#####

u, v, w = model.velocities
T, S = model.tracers.T, model.tracers.S

ubar = Field(Average(u, dims = (1,2)))
vbar = Field(Average(v, dims = (1,2)))
Tbar = Field(Average(T, dims = (1,2)))
Sbar = Field(Average(S, dims = (1,2)))

averaged_outputs = (; ubar, vbar, Tbar, Sbar)

#####
##### Build checkpointer and output writer
#####
simulation.output_writers[:jld2] = JLD2OutputWriter(model, averaged_outputs,
    filename = "NN_1D_channel_averages",
    schedule = TimeInterval(10minutes),
    overwrite_existing = true)

@info "Running the simulation..."

try
    run!(simulation, pickup = false)
catch err
    @info "run! threw an error! The error message is"
    showerror(stdout, err)
end

# #####
# ##### Visualization
# #####
#%%
using CairoMakie

ubar_data = FieldTimeSeries("./NN_1D_channel_averages.jld2", "ubar")
vbar_data = FieldTimeSeries("./NN_1D_channel_averages.jld2", "vbar")
Tbar_data = FieldTimeSeries("./NN_1D_channel_averages.jld2", "Tbar")
Sbar_data = FieldTimeSeries("./NN_1D_channel_averages.jld2", "Sbar")

zC = znodes(Tbar_data.grid, Center())
zF = znodes(Tbar_data.grid, Face())

Nt = length(Tbar_data.times)

fig = Figure(size = (1800, 600))
axu = CairoMakie.Axis(fig[1, 1], xlabel = "u (m s⁻¹)", ylabel = "z (m)")
axv = CairoMakie.Axis(fig[1, 2], xlabel = "v (m s⁻¹)", ylabel = "z (m)")
axT = CairoMakie.Axis(fig[1, 3], xlabel = "T (°C)", ylabel = "z (m)")
axS = CairoMakie.Axis(fig[1, 4], xlabel = "S (g kg⁻¹)", ylabel = "z (m)")
# slider = Slider(fig[2, :], range=1:Nt)
n = Observable(1)

ubarₙ = @lift interior(ubar_data[$n], 1, 1, :)
vbarₙ = @lift interior(vbar_data[$n], 1, 1, :)
Tbarₙ = @lift interior(Tbar_data[$n], 1, 1, :)
Sbarₙ = @lift interior(Sbar_data[$n], 1, 1, :)

ulim = (minimum(ubar_data), maximum(ubar_data))
vlim = (minimum(vbar_data), maximum(vbar_data))
Tlim = (minimum(Tbar_data), maximum(Tbar_data))
Slim = (minimum(Sbar_data), maximum(Sbar_data))

title_str = @lift "Time: $(round(Tbar_data.times[$n] / 86400, digits=3)) days"

lines!(axu, ubarₙ, zC)
lines!(axv, vbarₙ, zC)
lines!(axT, Tbarₙ, zC)
lines!(axS, Sbarₙ, zC)

xlims!(axu, ulim)
xlims!(axv, vlim)
xlims!(axT, Tlim)
xlims!(axS, Slim)

Label(fig[0, :], title_str, tellwidth = false)

CairoMakie.record(fig, "./NN_1D_fields.mp4", 1:Nt, framerate=60) do nn
    n[] = nn
end

# display(fig)
#%%