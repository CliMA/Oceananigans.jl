using GLMakie
using Oceananigans
using Oceananigans.Units
using Printf

using Oceananigans.TurbulenceClosures:
    RiBasedVerticalDiffusivity,
    CATKEVerticalDiffusivity,
    ConvectiveAdjustmentVerticalDiffusivity

#####
##### Choose a closure
#####

# closure = nothing
#closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), κ=1.0, ν=1.0)
closure = RiBasedVerticalDiffusivity()
#closure = CATKEVerticalDiffusivity()
#closure = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=0.1, convective_νz=0.1)

#####
##### Setup simulation
#####

grid = RectilinearGrid(size=32, z=(-256, 0), topology=(Flat, Flat, Bounded))
coriolis = FPlane(f=1e-4)

Qᵇ = +1e-7
Qᵘ = -1e-4

b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

model = HydrostaticFreeSurfaceModel(; grid, closure, coriolis,
                                    tracers = (:b, :e),
                                    buoyancy = BuoyancyTracer(),
                                    boundary_conditions = (; b=b_bcs, u=u_bcs))
                                    
N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model, b = bᵢ)

simulation = Simulation(model, Δt = 1minute, stop_time=48hours)

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                     schedule = TimeInterval(10minutes),
                     prefix = "windy_convection",
                     force = true)

progress(sim) = @info string("Iter: ", iteration(sim), " t: ", prettytime(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

run!(simulation)

#####
##### Visualize
#####

filepath = "windy_convection.jld2"
b_ts = FieldTimeSeries(filepath, "b")
u_ts = FieldTimeSeries(filepath, "u")
v_ts = FieldTimeSeries(filepath, "v")
z = znodes(b_ts)
Nt = length(b_ts.times)

fig = Figure(resolution=(1200, 800))

slider = Slider(fig[2, 1:2], range=1:Nt, startvalue=1)
n = slider.value

buoyancy_label = @lift "Buoyancy at t = " * prettytime(b_ts.times[$n])
velocities_label = @lift "Velocities at t = " * prettytime(b_ts.times[$n])
ax_b = Axis(fig[1, 1], xlabel=buoyancy_label, ylabel="z")
ax_u = Axis(fig[1, 2], xlabel=velocities_label, ylabel="z")

bn = @lift interior(b_ts[$n], 1, 1, :)
un = @lift interior(u_ts[$n], 1, 1, :)
vn = @lift interior(v_ts[$n], 1, 1, :)

lb = lines!(ax_b, bn, z)
lu = lines!(ax_u, un, z)
lv = lines!(ax_u, vn, z, linestyle=:dash)

xlims!(ax_b, -grid.Lz * N², 0)
xlims!(ax_u, -0.1, 0.1)

display(fig)

#=
Nframes = 100
Niters = 10 # per frame
record(fig, "ri_based_vertical_mixing.mp4", 1:Nframes, framerate=24) do frame
    [time_step!(simulation) for i = 1:Niters]
    update!()
end
=#
