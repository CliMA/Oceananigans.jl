using Printf
using Plots

using Oceananigans
using Oceananigans.OutputReaders: FieldTimeSeries 
using Oceananigans.Models
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary

experiment_name = "shallow_water_flow_past_cylinder"

underlying_grid = RegularRectilinearGrid(size=(128, 64), x=(-5, 10), y=(-3, 3), topology=(Periodic, Bounded, Flat), halo = (3, 3))
                              
cylinder(x, y, z)  = (x^2 + y^2) < 1
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(cylinder))

#=
damping_rate = 0.01 # relax fields on a 100 second time-scale
const x0 = -13.0 # center point of sponge
const dx = 1.0 # sponge width
smoothed_step_mask(x, y, z) = 1/2 * (1 + tanh((x - x0) / dx))

uh_sponge = Relaxation(rate = damping_rate, mask = smoothed_step_mask, target = 1)
 h_sponge = Relaxation(rate = damping_rate, mask = smoothed_step_mask, target = 1)

model = ShallowWaterModel(grid = grid, gravitational_acceleration = 1, forcing = (uh=uh_sponge, h=h_sponge))
=#

model = ShallowWaterModel(grid = grid, gravitational_acceleration = 1) #, forcing = (uh=uh_sponge, h=h_sponge))

set!(model, h = 1, uh = 1)

wall_clock = [time_ns()]

function progress(sim)
    @info(@sprintf("Iter: %d, time: %.2e, Δt: %.2e, wall time: %s, max|uh|: %.2f",
                   sim.model.clock.iteration,
                   sim.model.clock.time,
                   sim.Δt.Δt,
                   prettytime(1e-9 * (time_ns() - wall_clock[1])),
                   maximum(abs, sim.model.solution.uh)))

    wall_clock[1] = time_ns()

    return nothing
end

Δmin = min(grid.Δx, grid.Δy)

wizard = TimeStepWizard(cfl = 0.5, Δt = 0.01Δmin, max_change = 1.1, max_Δt = 0.05Δmin)

simulation = Simulation(model, Δt=wizard, stop_time=1, progress=progress, iteration_interval=10)

uh, vh, h = model.solution

ζ = ComputedField(∂x(vh / h) - ∂y(uh / h))

outputs = merge(model.solution, (ζ=ζ,))

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, outputs,
                     schedule = TimeInterval(0.1),
                     prefix = experiment_name,
                     field_slicer = nothing,
                     force = true)

run!(simulation)

filepath = experiment_name * ".jld2"

ζ_timeseries = FieldTimeSeries(filepath, "ζ")
u_timeseries = FieldTimeSeries(filepath, "uh")

grid = u_timeseries.grid

xζ, yζ, zζ = nodes(ζ_timeseries)
xu, yu, zu = nodes(u_timeseries)

anim = @animate for (n, t) in enumerate(ζ_timeseries.times)

    @info "    Plotting frame $n from iteration of $(length(ζ_timeseries.times))..."

    ζ = ζ_timeseries[n]
    u = u_timeseries[n]

    ζi = interior(ζ)[:, :, 1]
    ui = interior(u)[:, :, 1]

    clim = 5

    kwargs = Dict(:aspectratio => 1,
                  :linewidth => 0,
                  :colorbar => :none,
                  :ticks => nothing,
                  :clims => (-clim, clim),
                  :xlims => (-grid.Lx/2, grid.Lx/2),
                  :ylims => (-grid.Ly/2, grid.Ly/2))

    ζ_plot = heatmap(xζ, yζ, clamp.(ζi, -clim, clim)'; color = :balance, kwargs...)
    u_plot = heatmap(xu, yu, clamp.(ui, -clim, clim)'; color = :balance, kwargs...)

    ζ_title = @sprintf("ζ at t = %.1f", t)
    u_title = @sprintf("u at t = %.1f", t)

    plot(ζ_plot, u_plot, title = [ζ_title u_title], size = (4000, 2000))
end

mp4(anim, experiment_name * ".mp4", fps = 8)
