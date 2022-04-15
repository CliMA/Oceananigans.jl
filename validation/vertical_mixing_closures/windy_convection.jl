using GLMakie
using Oceananigans
using Oceananigans.Units
using Printf

using Oceananigans.TurbulenceClosures:
    RiBasedVerticalDiffusivity,
    CATKEVerticalDiffusivity,
    ConvectiveAdjustmentVerticalDiffusivity,
    ExplicitTimeDiscretization

#####
##### Setup simulation
#####

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=0.1, convective_νz=0.01)

grid = RectilinearGrid(size=32, z=(-256, 0), topology=(Flat, Flat, Bounded))
coriolis = FPlane(f=1e-4)

N² = 1e-5
Qᵇ = +1e-8
Qᵘ = -1e-3

b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

closures_to_run = [
                   CATKEVerticalDiffusivity(),
                   RiBasedVerticalDiffusivity(),
                   convective_adjustment,
                   ]   

for closure in closures_to_run

    model = HydrostaticFreeSurfaceModel(; grid, closure, coriolis,
                                        tracers = (:b, :e),
                                        buoyancy = BuoyancyTracer(),
                                        boundary_conditions = (; b=b_bcs, u=u_bcs))
                                        
    bᵢ(x, y, z) = N² * z
    set!(model, b = bᵢ)

    simulation = Simulation(model, Δt=10minute, stop_time=48hours)

    closurename = string(nameof(typeof(closure)))

    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                         schedule = TimeInterval(10minutes),
                         prefix = "windy_convection_" * closurename,
                         overwrite_existing = true)

    progress(sim) = @info string("Iter: ", iteration(sim), " t: ", prettytime(sim))
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    @info "Running a simulation of $model..."

    run!(simulation)
end

#####
##### Visualize
#####

b_ts = []
u_ts = []
v_ts = []
e_ts = []

for closure in closures_to_run
    closurename = string(nameof(typeof(closure)))
    filepath = "windy_convection_" * closurename * ".jld2"

    push!(b_ts, FieldTimeSeries(filepath, "b"))
    push!(u_ts, FieldTimeSeries(filepath, "u"))
    push!(v_ts, FieldTimeSeries(filepath, "v"))
    push!(e_ts, FieldTimeSeries(filepath, "e"))
end

b1 = first(b_ts)

z = znodes(b1)
Nt = length(b1.times)

fig = Figure(resolution=(1200, 800))

slider = Slider(fig[2, 1:2], range=1:Nt, startvalue=1)
n = slider.value

buoyancy_label = @lift "Buoyancy at t = " * prettytime(b1.times[$n])
velocities_label = @lift "Velocities at t = " * prettytime(b1.times[$n])
TKE_label = @lift "Turbulent kinetic energy t = " * prettytime(b1.times[$n])
ax_b = Axis(fig[1, 1], xlabel=buoyancy_label, ylabel="z")
ax_u = Axis(fig[1, 2], xlabel=velocities_label, ylabel="z")
ax_e = Axis(fig[1, 3], xlabel=TKE_label, ylabel="z")

xlims!(ax_b, -grid.Lz * N², 0)
xlims!(ax_u, -1.0, 1.0)
xlims!(ax_e, -1e-4, 3e-3)

colors = [:black, :blue, :red, :orange]

for (i, closure) in enumerate(closures_to_run)
    bn = @lift interior(b_ts[i][$n], 1, 1, :)
    un = @lift interior(u_ts[i][$n], 1, 1, :)
    vn = @lift interior(v_ts[i][$n], 1, 1, :)
    en = @lift interior(e_ts[i][$n], 1, 1, :)
    
    closurename = string(nameof(typeof(closure)))

    lines!(ax_b, bn, z, label=closurename, color=colors[i])
    lines!(ax_u, un, z, label="u, " * closurename, color=colors[i])
    lines!(ax_u, vn, z, label="v, " * closurename, linestyle=:dash, color=colors[i])
    lines!(ax_e, en, z, label="e, " * closurename, linestyle=:dash, color=colors[i])
end

axislegend(ax_b, position=:lb)
axislegend(ax_u, position=:rb)
axislegend(ax_e, position=:rb)

display(fig)

record(fig, "windy_convection.mp4", 1:Nt, framerate=24) do nn
    n[] = nn
end

