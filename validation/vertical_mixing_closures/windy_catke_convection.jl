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
Qᵇ = +1.2e-7
Qᵘ = -1e-4 #

b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

closure = CATKEVerticalDiffusivity()

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
                     filename = "windy_convection_catke",
                     overwrite_existing = true)

progress(sim) = @info string("Iter: ", iteration(sim), " t: ", prettytime(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

@info "Running a simulation of $model..."

run!(simulation)

#####
##### Visualize
#####

filepath = "windy_convection_catke.jld2"

b_ts = FieldTimeSeries(filepath, "b")
u_ts = FieldTimeSeries(filepath, "u")
v_ts = FieldTimeSeries(filepath, "v")
e_ts = FieldTimeSeries(filepath, "e")

z = znodes(b_ts)
Nt = length(b_ts.times)

fig = Figure(resolution=(1200, 800))

slider = Slider(fig[2, 1:2], range=1:Nt, startvalue=1)
n = slider.value

buoyancy_label = @lift "Buoyancy at t = " * prettytime(b_ts.times[$n])
velocities_label = @lift "Velocities at t = " * prettytime(b_ts.times[$n])
TKE_label = @lift "Turbulent kinetic energy t = " * prettytime(b_ts.times[$n])
ax_b = Axis(fig[1, 1], xlabel=buoyancy_label, ylabel="z")
ax_u = Axis(fig[1, 2], xlabel=velocities_label, ylabel="z")
ax_e = Axis(fig[1, 3], xlabel=TKE_label, ylabel="z")

xlims!(ax_b, -grid.Lz * N², 0)
xlims!(ax_u, -1.0, 1.0)
xlims!(ax_e, -1e-4, 3e-3)

colors = [:black, :blue, :red, :orange]

bn = @lift interior(b_ts[$n], 1, 1, :)
un = @lift interior(u_ts[$n], 1, 1, :)
vn = @lift interior(v_ts[$n], 1, 1, :)
en = @lift interior(e_ts[$n], 1, 1, :)

lines!(ax_b, bn, z)
lines!(ax_u, un, z, label="u")
lines!(ax_u, vn, z, label="v", linestyle=:dash)
lines!(ax_e, en, z)

xlims!(ax_u, -0.03, 0.03)
xlims!(ax_e, -0.0001, 0.0005)

axislegend(ax_u, position=:rb)

display(fig)

record(fig, "windy_catke_convection.mp4", 1:Nt, framerate=24) do nn
    n[] = nn
end

