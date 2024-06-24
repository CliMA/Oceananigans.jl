#using Pkg
# pkg"add Oceananigans CairoMakie"
using Oceananigans
include("NN_closure.jl")
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

const Ly = 2000kilometers # meridional domain length [m]

# Architecture
model_architecture = CPU()

# number of grid points
Ny = 192
Nz = 128

const Lz = 1024

grid = RectilinearGrid(model_architecture,
    topology = (Flat, Bounded, Bounded),
    size = (Ny, Nz),
    halo = (3, 3),
    y = (0, Ly),
    z = (-Lz, 0))

@info "Built a grid: $grid."

#####
##### Boundary conditions
#####
T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(1e-4), bottom=FluxBoundaryCondition(0.0))

#####
##### Coriolis
#####

const f₀ = 8e-5
const β = 1e-11
coriolis = BetaPlane(f₀=f₀, β = β)

#####
##### Forcing and initial condition
#####
const dTdz = 0.014
const dSdz = 0.0021

const T_surface = 20.0
const S_surface = 36.6

T_initial(y, z) = dTdz * z + T_surface
S_initial(y, z) = dSdz * z + S_surface

# closure
κh = 0.5e-5 # [m²/s] horizontal diffusivity
νh = 30.0   # [m²/s] horizontal viscocity
κz = 0.5e-5 # [m²/s] vertical diffusivity
νz = 3e-4   # [m²/s] vertical viscocity

horizontal_closure = HorizontalScalarDiffusivity(ν = νh, κ = κh)
vertical_closure = VerticalScalarDiffusivity(ν = νz, κ = κz)

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
    convective_νz = 0.0)

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
    # closure = (horizontal_closure, vertical_closure, convective_adjustment),
    tracers = (:T, :S),
    boundary_conditions = (; T = T_bcs),
    # forcing = (; b = Fb)
)

@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
noise(y, z) = rand() * exp(z / 8)

T_initial_noisy(y, z) = T_initial(y, z) + 1e-6 * noise(y, z)
S_initial_noisy(y, z) = S_initial(y, z) + 1e-6 * noise(y, z)

set!(model, T=T_initial_noisy, S=S_initial_noisy)
using Oceananigans.TimeSteppers: update_state!
update_state!(model)
#####
##### Simulation building
#####
Δt₀ = 5minutes
stop_time = 1days

simulation = Simulation(model, Δt = Δt₀, stop_time = stop_time)

# add timestep wizard callback
# wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=20minutes)
# simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

# add progress callback
wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
        100 * (sim.model.clock.time / sim.stop_time),
        sim.model.clock.iteration,
        prettytime(sim.model.clock.time),
        prettytime(1e-9 * (time_ns() - wall_clock[1])),
        maximum(abs, sim.model.velocities.u),
        maximum(abs, sim.model.velocities.v),
        maximum(abs, sim.model.velocities.w),
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

ζ = Field(∂x(v) - ∂y(u))

Tbar = Field(Average(T, dims = 1))
Sbar = Field(Average(S, dims = 1))
V = Field(Average(v, dims = 1))
W = Field(Average(w, dims = 1))

T′ = T - Tbar
S′ = S - Sbar
v′ = v - V
w′ = w - W

v′T′ = Field(Average(v′ * T′, dims = 1))
w′T′ = Field(Average(w′ * T′, dims = 1))
v′S′ = Field(Average(v′ * S′, dims = 1))
w′S′ = Field(Average(w′ * S′, dims = 1))

outputs = (; T, S, ζ, w)

averaged_outputs = (; v′T′, w′T′, v′S′, w′S′, Tbar, Sbar)

#####
##### Build checkpointer and output writer
#####

simulation.output_writers[:checkpointer] = Checkpointer(model,
    schedule = TimeInterval(100days),
    prefix = "NN_channel",
    overwrite_existing = true)

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
    schedule = TimeInterval(5days),
    filename = "NN_channel",
    # field_slicer = nothing,
    verbose = true,
    overwrite_existing = true)

simulation.output_writers[:averages] = JLD2OutputWriter(model, averaged_outputs,
    schedule = AveragedTimeInterval(1days, window = 1days, stride = 1),
    filename = "NN_channel_averages",
    verbose = true,
    overwrite_existing = true)

@info "Running the simulation..."

try
    run!(simulation, pickup = false)
catch err
    @info "run! threw an error! The error message is"
    showerror(stdout, err)
end

#=
# #####
# ##### Visualization
# #####

using Plots

grid = RectilinearGrid(architecture = CPU(),
    topology = (Periodic, Bounded, Bounded),
    size = (grid.Nx, grid.Ny, grid.Nz),
    halo = (3, 3, 3),
    x = (0, grid.Lx),
    y = (0, grid.Ly),
    z = z_faces)

xζ, yζ, zζ = nodes((Face, Face, Center), grid)
xc, yc, zc = nodes((Center, Center, Center), grid)
xw, yw, zw = nodes((Center, Center, Face), grid)

j′ = round(Int, grid.Ny / 2)
y′ = yζ[j′]

b_timeseries = FieldTimeSeries("abernathey_channel.jld2", "b", grid = grid)
ζ_timeseries = FieldTimeSeries("abernathey_channel.jld2", "ζ", grid = grid)
w_timeseries = FieldTimeSeries("abernathey_channel.jld2", "w", grid = grid)

@show b_timeseries

anim = @animate for i in 1:length(b_timeseries.times)
    b = b_timeseries[i]
    ζ = ζ_timeseries[i]
    w = w_timeseries[i]

    b′ = interior(b) .- mean(b)
    b_xy = b′[:, :, grid.Nz]
    ζ_xy = interior(ζ)[:, :, grid.Nz]
    ζ_xz = interior(ζ)[:, j′, :]
    w_xz = interior(w)[:, j′, :]

    @show bmax = max(1e-9, maximum(abs, b_xy))
    @show ζmax = max(1e-9, maximum(abs, ζ_xy))
    @show wmax = max(1e-9, maximum(abs, w_xz))

    blims = (-bmax, bmax) .* 0.8
    ζlims = (-ζmax, ζmax) .* 0.8
    wlims = (-wmax, wmax) .* 0.8

    blevels = vcat([-bmax], range(blims[1], blims[2], length = 31), [bmax])
    ζlevels = vcat([-ζmax], range(ζlims[1], ζlims[2], length = 31), [ζmax])
    wlevels = vcat([-wmax], range(wlims[1], wlims[2], length = 31), [wmax])

    xlims = (-grid.Lx / 2, grid.Lx / 2) .* 1e-3
    ylims = (0, grid.Ly) .* 1e-3
    zlims = (-grid.Lz, 0)

    w_xz_plot = contourf(xw * 1e-3, zw, w_xz',
        xlabel = "x (km)",
        ylabel = "z (m)",
        aspectratio = 0.05,
        linewidth = 0,
        levels = wlevels,
        clims = wlims,
        xlims = xlims,
        ylims = zlims,
        color = :balance)

    ζ_xy_plot = contourf(xζ * 1e-3, yζ * 1e-3, ζ_xy',
        xlabel = "x (km)",
        ylabel = "y (km)",
        aspectratio = :equal,
        linewidth = 0,
        levels = ζlevels,
        clims = ζlims,
        xlims = xlims,
        ylims = ylims,
        color = :balance)

    b_xy_plot = contourf(xc * 1e-3, yc * 1e-3, b_xy',
        xlabel = "x (km)",
        ylabel = "y (km)",
        aspectratio = :equal,
        linewidth = 0,
        levels = blevels,
        clims = blims,
        xlims = xlims,
        ylims = ylims,
        color = :balance)

    w_xz_title = @sprintf("w(x, z) at t = %s", prettytime(ζ_timeseries.times[i]))
    ζ_xz_title = @sprintf("ζ(x, z) at t = %s", prettytime(ζ_timeseries.times[i]))
    ζ_xy_title = "ζ(x, y)"
    b_xy_title = "b(x, y)"

    layout = @layout [upper_slice_plot{0.2h}
        Plots.grid(1, 2)]

    plot(w_xz_plot, ζ_xy_plot, b_xy_plot, layout = layout, size = (1200, 1200), title = [w_xz_title ζ_xy_title b_xy_title])
end

mp4(anim, "abernathey_channel.mp4", fps = 8) #hide
=#