#using Pkg
# pkg"add Oceananigans CairoMakie"
using Oceananigans
include("NN_closure_global_nof_BBLRifirstzone510.jl")
include("xin_kai_vertical_diffusivity_local_2step.jl")
include("feature_scaling.jl")

ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Statistics

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.TimeSteppers: update_state!
using SeawaterPolynomials
using SeawaterPolynomials:TEOS10


# Architecture
model_architecture = CPU()

file = jldopen("model_inference_run_nof_BBLRifirstzone510.jld2", "r")

# number of grid points
const Nz = file["Nz"]
const Lz = file["Lz"]

grid = RectilinearGrid(model_architecture,
    topology = (Flat, Flat, Bounded),
    size = Nz,
    halo = 3,
    z = (-Lz, 0))

@info "Built a grid: $grid."

#####
##### Boundary conditions
#####
const dTdz = file["dTdz"]
const dSdz = file["dSdz"]

const T_surface = file["T_surface"]
const S_surface = file["S_surface"]

T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(file["wT_top"]))
S_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(file["wS_top"]))

#####
##### Coriolis
#####

const f₀ = file["f₀"]
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
    closure = (base_closure, nn_closure),
    tracers = (:T, :S),
    boundary_conditions = (; T = T_bcs),
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
update_state!(model)
#####
##### Simulation building
#####
Δt₀ = file["Δt"]
stop_time = file["τ"]

simulation = Simulation(model, Δt = Δt₀, stop_time = stop_time)

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
wT_residual, wS_residual = model.diffusivity_fields[2].wT, model.diffusivity_fields[2].wS
ν, κ = model.diffusivity_fields[1].κᵘ, model.diffusivity_fields[1].κᶜ

Tbar = Field(Average(T, dims = (1,2)))
Sbar = Field(Average(S, dims = (1,2)))

averaged_outputs = (; Tbar, Sbar, wT_residual, wS_residual, ν, κ)

#####
##### Build checkpointer and output writer
#####
simulation.output_writers[:jld2] = JLD2OutputWriter(model, averaged_outputs,
    filename = "NN_1D_channel_averages_nof_BBLRifirstzone510",
    schedule = TimeInterval(Δt₀),
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
using GLMakie

Tbar_data = FieldTimeSeries("./NN_1D_channel_averages_nof_BBLRifirstzone510.jld2", "Tbar")
Sbar_data = FieldTimeSeries("./NN_1D_channel_averages_nof_BBLRifirstzone510.jld2", "Sbar")
wT_residual_data = FieldTimeSeries("./NN_1D_channel_averages_nof_BBLRifirstzone510.jld2", "wT_residual")
wS_residual_data = FieldTimeSeries("./NN_1D_channel_averages_nof_BBLRifirstzone510.jld2", "wS_residual")
ν_data = FieldTimeSeries("./NN_1D_channel_averages_nof_BBLRifirstzone510.jld2", "ν")
κ_data = FieldTimeSeries("./NN_1D_channel_averages_nof_BBLRifirstzone510.jld2", "κ")

#%%
zC = znodes(Tbar_data.grid, Center())
zF = znodes(Tbar_data.grid, Face())

Nt = length(Tbar_data.times)

fig = Figure(size = (1500, 1000))
axT = GLMakie.Axis(fig[1, 1], xlabel = "T (°C)", ylabel = "z (m)")
axS = GLMakie.Axis(fig[2, 1], xlabel = "S (g kg⁻¹)", ylabel = "z (m)")
axwT_residual = GLMakie.Axis(fig[1, 2], xlabel = "wT residual", ylabel = "z (m)")
axwS_residual = GLMakie.Axis(fig[2, 2], xlabel = "wS residual", ylabel = "z (m)")
axν = GLMakie.Axis(fig[1, 3], xlabel = "ν (m² s⁻¹)", ylabel = "z (m)", xscale=log10)
axκ = GLMakie.Axis(fig[2, 3], xlabel = "κ (m² s⁻¹)", ylabel = "z (m)", xscale=log10)

slider = Slider(fig[3, :], range=2:Nt)
n = slider.value

Tbarₙ = @lift interior(Tbar_data[$n], 1, 1, :)
Sbarₙ = @lift interior(Sbar_data[$n], 1, 1, :)
wT_residualₙ = @lift interior(wT_residual_data[$n], 1, 1, :)
wS_residualₙ = @lift interior(wS_residual_data[$n], 1, 1, :)
νₙ = @lift interior(ν_data[$n], 1, 1, 2:32)
κₙ = @lift interior(κ_data[$n], 1, 1, 2:32)

Tbar_truthₙ = @lift file["sol_T"][:, $n]
Sbar_truthₙ = @lift file["sol_S"][:, $n]
wT_residual_truthₙ = @lift file["sol_wT_residual_unscaled"][:, $n]
wS_residual_truthₙ = @lift file["sol_wS_residual_unscaled"][:, $n]
ν_truthₙ = @lift file["sol_ν"][2:32, $n]
κ_truthₙ = @lift file["sol_κ"][2:32, $n]

title_str = @lift "Time: $(round(Tbar_data.times[$n] / 86400, digits=3)) days"

wTlim = (minimum(interior(wT_residual_data)), maximum(interior(wT_residual_data)))
wSlim = (minimum(interior(wS_residual_data)), maximum(interior(wS_residual_data)))

νlim = (1e-6, 10)
κlim = (1e-6, 10)

lines!(axT, Tbarₙ, zC, label="Oceananigans")
lines!(axS, Sbarₙ, zC, label="Oceananigans")

lines!(axwT_residual, wT_residualₙ, zF, label="Oceananigans")
lines!(axwS_residual, wS_residualₙ, zF, label="Oceananigans")

lines!(axν, νₙ, zF[2:32], label="Oceananigans")
lines!(axκ, κₙ, zF[2:32], label="Oceananigans")

lines!(axT, Tbar_truthₙ, zC, label="Truth")
lines!(axS, Sbar_truthₙ, zC, label="Truth")

lines!(axwT_residual, wT_residual_truthₙ, zF, label="Truth")
lines!(axwS_residual, wS_residual_truthₙ, zF, label="Truth")

lines!(axν, ν_truthₙ, zF[2:32], label="Truth")
lines!(axκ, κ_truthₙ, zF[2:32], label="Truth")

xlims!(axwT_residual, wTlim)
xlims!(axwS_residual, wSlim)
xlims!(axν, νlim)
xlims!(axκ, κlim)

linkyaxes!(axT, axS, axwT_residual, axwS_residual, axν, axκ)

axislegend(axT, position = :lb)
Label(fig[0, :], title_str, tellwidth = false)

# GLMakie.record(fig, "./NN_1D_validation_nof_BBL.mp4", 1:Nt, framerate=60, px_per_unit=4) do nn
#     @info nn
#     n[] = nn
# end

display(fig)
#%%
close(file)