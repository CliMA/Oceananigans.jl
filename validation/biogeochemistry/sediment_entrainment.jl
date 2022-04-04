using Oceananigans
using Oceananigans.Units
using GLMakie
using Printf

grid = RectilinearGrid(size=(64, 128), halo=(3, 3),
                       x=(0, 64), z=(0, 32),
                       topology=(Periodic, Flat, Bounded))

@inline u_drag(x, y, t, u, w, Cᵈ) = - Cᵈ * u * sqrt(u^2 + w^2)
u_bottom_bc = FluxBoundaryCondition(u_drag, field_dependencies=(:u, :w), parameters=1e-3)
u_bcs = FieldBoundaryConditions(bottom=u_bottom_bc)

sediment_bottom_bc = ValueBoundaryCondition(1)
sediment_bcs = FieldBoundaryConditions(bottom=sediment_bottom_bc)

r_sediment = 1e-4 # "Fine sand"
ρ_sediment = 1400 # kg m⁻³
ρ_ocean = 1026 # kg m⁻³
Δb = - 9.81 * (ρ_sediment - ρ_ocean) / ρ_ocean
ν_molecular = 1.05e-6
@show w_sediment = 2/9 * Δb / ν_molecular * r_sediment^2
sinking = AdvectiveForcing(WENO5(), w=w_sediment)

model = NonhydrostaticModel(; grid,
                            advection = WENO5(),
                            timestepper = :RungeKutta3,
                            tracers = (:b, :sediment),
                            buoyancy = BuoyancyTracer(),
                            closure = ScalarDiffusivity(ν=1e-5, κ=(b=1e-5, sediment=1e-3)),
                            forcing = (; sediment = sinking),
                            boundary_conditions = (; u=u_bcs, sediment=sediment_bcs))

bᵢ(x, y, z) = 1e-4 * z
uᵢ(x, y, z) = 1 - tanh(z / 4)
wᵢ(x, y, z) = 1e-3 * rand() * exp(-z^2 / 2)
sᵢ(x, y, z) = 0.1 * exp(-((x - 32)^2 + (z - 16)^2) / 8)
set!(model, b=bᵢ, u=uᵢ, w=wᵢ, sediment=sᵢ)

Δz = grid.Δzᵃᵃᶜ
Δt = 0.2 * Δz / abs(w_sediment)
simulation = Simulation(model; Δt, stop_time=2hours)

wizard = TimeStepWizard(cfl=0.5, max_Δt=Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

wall_clock = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    @info @sprintf("Iter: %d, time: %s, Δt: %s, min(u): %.2e, min(s): %.2e, wall time: %s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   minimum(model.velocities.u), minimum(model.tracers.sediment),
                   prettytime(elapsed))

    wall_clock[] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

fig = Figure(resolution=(1600, 600))
ax_ξ = Axis(fig[1, 1])
ax_w = Axis(fig[1, 2])
ax_s = Axis(fig[1, 3])

sediment = model.tracers.sediment
u, v, w = model.velocities

ξ = compute!(Field(∂z(u) - ∂x(w)))

wlim = maximum(abs, w) / 2
ξlim = maximum(abs, ξ) / 2
smax = maximum(sediment)

hm_ξ = heatmap!(ax_ξ, interior(ξ, :, 1, :), colormap=:redblue, colorrange=(-wlim, wlim))
hm_w = heatmap!(ax_w, interior(w, :, 1, :), colormap=:redblue, colorrange=(-wlim, wlim))
hm_s = heatmap!(ax_s, interior(sediment, :, 1, :), colormap=:haline, colorrange=(0, smax))

display(fig)

function update_plot!(sim)
    compute!(ξ)
    wlim = maximum(abs, w) / 2
    ξlim = maximum(abs, ξ) / 2
    smax = maximum(sediment)

    hm_w.attributes.colorrange[] = (-wlim, wlim)
    hm_ξ.attributes.colorrange[] = (-ξlim, ξlim)
    # hm_s.attributes.colorrange[] = (0, smax/2)

    hm_s.input_args[1][] = interior(sediment, :, 1, :)
    hm_w.input_args[1][] = interior(w, :, 1, :)
    hm_ξ.input_args[1][] = interior(ξ, :, 1, :)

    return nothing
end

simulation.callbacks[:plot] = Callback(update_plot!, IterationInterval(100))

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, (; ξ)),
                     schedule = TimeInterval(1.0),
                     with_halos = false,
                     prefix = "sediment_entrainment",
                     force = true)

run!(simulation)

#####
##### Visualize
#####

filepath = "sediment_entrainment.jld2"
ξt = FieldTimeSeries(filepath, "ξ")
st = FieldTimeSeries(filepath, "sediment")
wt = FieldTimeSeries(filepath, "w")
Nt = length(wt.times)

fig = Figure(resolution=(1600, 600))
ax_ξ = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)", title="Vorticity", aspect=2)
ax_w = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)", title="Vertical velocity", aspect=2)
ax_s = Axis(fig[1, 3], xlabel="x (m)", ylabel="z (m)", title="Sediment concentration", aspect=2)
slider = Slider(fig[2, :], range=1:Nt, startvalue=1)
n = slider.value

title = @lift string("Sediment entrainment at t = ", prettytime(wt.times[$n]))
Label(fig[0, :], title)

xξ, yξ, zξ = nodes(ξt)
xw, yw, zw = nodes(wt)
xs, ys, zs = nodes(st)

ξn = @lift interior(ξt[$n], :, 1, :)
sn = @lift interior(st[$n], :, 1, :)
wn = @lift interior(wt[$n], :, 1, :)

wlim = maximum(abs, wt[Nt]) / 2
ξlim = maximum(abs, ξt[Nt]) / 2
slim = maximum(st[Nt]) / 4

hm_ξ = heatmap!(ax_ξ, xξ, zξ, ξn, colormap=:redblue, colorrange=(-wlim, wlim))
hm_w = heatmap!(ax_w, xw, zw, wn, colormap=:redblue, colorrange=(-wlim, wlim))
hm_s = heatmap!(ax_s, xs, zs, sn, colormap=:haline, colorrange=(0, slim))

display(fig)

record(fig, "sediment_entrainment.mp4", 1:Nt, framerate=160) do nn
    n[] = nn
end

