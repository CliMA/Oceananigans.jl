using Printf
using Statistics
using Random

using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization 
using Oceananigans.TurbulenceClosures.MEWSVerticalDiffusivities: MEWSVerticalDiffusivity

# Domain
Ny = 64
Nz = 32
Ly = 2000kilometers  # north-south extent [m]
Lz = 1kilometers     # depth [m]
Δy = 100kilometers
N² = 1e-5 # [s⁻²] buoyancy frequency / stratification
M² = 1e-7 # [s⁻²] horizontal buoyancy gradient
Δb = Δy * M²

save_interval = 1day
stop_time = 30days

# We choose a regular grid though because of numerical issues that yet need to be resolved
grid = RectilinearGrid(CPU();
                       topology = (Flat, Bounded, Bounded), 
                       size = (Ny, Nz), 
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0),
                       halo = (4, 4))

coriolis = BetaPlane(latitude = -45)

#vitd = VerticallyImplicitTimeDiscretization()
#mesoscale_closure = VerticalScalarDiffusivity(vitd, ν=1e0)
mesoscale_closure = MEWSVerticalDiffusivity(Cᴷ=1e-1, Cⁿ=10.0, Cᴰ=1e-3, Cʰ=1)

model = HydrostaticFreeSurfaceModel(; grid, coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    closure = mesoscale_closure,
                                    tracers = (:b, :K),
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO(),
                                    free_surface = ImplicitFreeSurface())

@info "Built $model."

# Baroclinically unstable initial condition
f = coriolis.f₀
ramp(y, Δy) = (1 + tanh(y / Δy)) / 2
d_ramp_dy(y, Δy) = sech(y / Δy)^2 / (2Δy)

bᵢ(x, y, z) =   Δb * ramp(y, Δy) + N² * z
uᵢ(x, y, z) = - Δb / f * d_ramp_dy(y, Δy) * (z + Lz/2)

set!(model, b=bᵢ, u=uᵢ, K=1e-4)

simulation = Simulation(model; Δt=20minutes, stop_time)

νₑ = model.diffusivity_fields.νₑ
νₖ = model.diffusivity_fields.νₖ

# add progress callback
wall_clock = Ref(time_ns())

function print_progress(sim)
    msg1 = @sprintf("i: % 4d, t: % 12s, wall time: % 12s, extrema(νₑ): (%6.3e, %6.3e) m² s⁻¹, extrema(νₖ): (%6.3e, %6.3e) m² s⁻¹, ",
                    iteration(sim),
                    prettytime(sim),
                    prettytime(1e-9 * (time_ns() - wall_clock[])),
                    maximum(νₑ),
                    minimum(νₑ),
                    maximum(νₖ),
                    minimum(νₖ))

     msg2 = @sprintf(" extrema(K): (%6.3e, %6.3e) m² s⁻², max|u|: (%6.3e, %6.3e, %6.3e) m s⁻¹",
                     maximum(sim.model.tracers.K),
                     minimum(sim.model.tracers.K),
                     maximum(abs, sim.model.velocities.u),
                     maximum(abs, sim.model.velocities.v),
                     maximum(abs, sim.model.velocities.w))

    @info msg1 * msg2

    wall_clock[] = time_ns()
    
    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(10))

u, v, w = model.velocities
outputs = merge(model.velocities, model.tracers, (; νₑ, νₖ, uz=∂z(u)))

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      #schedule = TimeInterval(save_interval),
                                                      schedule = IterationInterval(10),
                                                      filename = "zonally_averaged_baroclinic_adjustment",
                                                      overwrite_existing = true)

@info "Running the simulation..."

run!(simulation)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)

#####
##### Visualize
#####

using GLMakie
using Oceananigans

fig = Figure(resolution = (2400, 1200))

filepath = "zonally_averaged_baroclinic_adjustment.jld2"

ut = FieldTimeSeries(filepath, "u")
uzt = FieldTimeSeries(filepath, "uz")
bt = FieldTimeSeries(filepath, "b")
Kt = FieldTimeSeries(filepath, "K")
νₑt = FieldTimeSeries(filepath, "νₑ")
νₖt = FieldTimeSeries(filepath, "νₖ")

times = bt.times
grid = bt.grid
Nt = length(times)

slider = Slider(fig[1, 1], range=1:Nt, startvalue=1)
n = slider.value

x, y, z = nodes((Center, Center, Center), grid)

bn = @lift interior(bt[$n], 1, :, :)
un = @lift interior(ut[$n], 1, :, :)
uzn = @lift interior(uzt[$n], 1, :, :)
Kn = @lift interior(Kt[$n], 1, :, :)
νₑn = @lift interior(νₑt[$n], 1, :, :)
νₖn = @lift interior(νₖt[$n], 1, :, :)

ulim = maximum(abs, ut) / 4
uzlim = maximum(abs, uzt) / 10
Klim = maximum(abs, Kt) / 2

x, y, z = nodes(bt)
xz, yz, zz = nodes(uzt)

x = x ./ 1e3
y = y ./ 1e3

xz = xz ./ 1e3
yz = yz ./ 1e3

titlestr = @lift string("Zonal velocity at ", prettytime(times[$n]))

axu = Axis(fig[2, 1])
hm = heatmap!(axu, y, z, un, colorrange=(-ulim, ulim), colormap=:balance)
contour!(axu, y, z, bn, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[2, 2], hm, label="Zonally-averaged zonal velocity (m s⁻¹)")

axuz = Axis(fig[3, 1])
hm = heatmap!(axuz, yz, zz, uzn, colorrange=(-uzlim, uzlim), colormap=:balance)
contour!(axuz, y, z, bn, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[3, 2], hm, label="Zonally-averaged shear (s⁻¹)")

axk = Axis(fig[4, 1])
hm = heatmap!(axk, y, z, Kn, colorrange=(Klim/10, Klim), colormap=:solar)
contour!(axk, y, z, bn, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[4, 2], hm, label="Zonally-averaged eddy kinetic energy (m² s⁻²)")

#display(fig)

# record(fig, filename * ".mp4", 1:Nt, framerate=8) do i
#     @info "Plotting frame $i of $Nt"
#     n[] = i
# end

