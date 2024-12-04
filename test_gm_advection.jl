using Printf
using Statistics
using Random
using Oceananigans
using Oceananigans.Units
using GLMakie
using Oceananigans.TurbulenceClosures: MesoscaleEddyTransport, IsopycnalSkewSymmetricDiffusivity

filename = "coarse_baroclinic_adjustment"

# Domain
Ly = 1000kilometers  # north-south extent [m]
Lz = 1kilometers     # depth [m]
Ny = 20
Nz = 20
save_fields_interval = 2day
stop_time = 300days
Δt = 10minutes

grid = RectilinearGrid(topology = (Flat, Bounded, Bounded),
                       size = (Ny, Nz), 
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0),
                       halo = (4, 4))

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(y -> y > 0 ? -Lz : -Lz/2))

coriolis = FPlane(latitude = -45)

@info "Building a model..."

adv_closure = MesoscaleEddyTransport()
cox_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew = adv_closure.κ, 
                                                isopycnal_tensor = adv_closure.isopycnal_tensor,
                                                slope_limiter = adv_closure.slope_limiter)

model = HydrostaticFreeSurfaceModel(; grid, coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    tracer_advection = WENO(),
                                    closure = adv_closure,
                                    tracers = (:b, :c))

@info "Built $model."

"""
Linear ramp from 0 to 1 between -Δy/2 and +Δy/2.

For example:

y < y₀           => ramp = 0
y₀ < y < y₀ + Δy => ramp = y / Δy
y > y₀ + Δy      => ramp = 1
"""
ramp(y, Δ) = min(max(0, y / Δ + 1/2), 1)

# Parameters
N² = 4e-6 # [s⁻²] buoyancy frequency / stratification
M² = 8e-8 # [s⁻²] horizontal buoyancy gradient

Δy = 100kilometers
Δz = 100

Δc = 2Δy
Δb = Δy * M²
ϵb = 1e-2 * Δb # noise amplitude

bᵢ(y, z) = N² * z + Δb * ramp(y, Δy)
cᵢ(y, z) = exp(-y^2 / 2Δc^2) * exp(-(z + Lz/4)^2 / 2Δz^2)

set!(model, b=bᵢ, c=cᵢ)

#####
##### Simulation building
#####

simulation = Simulation(model; Δt, stop_time)

wall_clock = Ref(time_ns())

function progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            prettytime(sim.Δt))

    wall_clock[] = time_ns()
    
    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

# eddy_velocities = (ue = model.diffusivity_fields.u, ve = model.diffusivity_fields.v, we = model.diffusivity_fields.w)

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers), # , eddy_velocities),
                                                      schedule = TimeInterval(save_fields_interval),
                                                      filename = filename * "_fields",
                                                      overwrite_existing = true)

@info "Running the simulation..."

run!(simulation)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)

#####
##### Visualize
#####

fig = Figure(size=(1400, 700))

filepath = filename * "_fields.jld2"

ut = FieldTimeSeries(filepath, "u")
# ue = FieldTimeSeries(filepath, "ue")
bt = FieldTimeSeries(filepath, "b")
ct = FieldTimeSeries(filepath, "c")

# Build coordinates, rescaling the vertical coordinate
x, y, z = nodes(bt)

zscale = 1
z = z .* zscale

#####
##### Plot buoyancy...
#####

times = bt.times
Nt = length(times)

un(n) = interior(mean(ut[n], dims=1), 1, :, :) # .+ interior(mean(ue[n], dims=1), 1, :, :) 
bn(n) = interior(mean(bt[n], dims=1), 1, :, :)
cn(n) = interior(mean(ct[n], dims=1), 1, :, :)

@show min_c = 0
@show max_c = 1
@show max_u = max([maximum(abs, un(n)) for n in 1:Nt]...) * 0.5
min_u = - max_u

axu = Axis(fig[2, 1], xlabel="$gradient (km)", ylabel="z (km)", title="Zonal velocity")
axc = Axis(fig[3, 1], xlabel="$gradient (km)", ylabel="z (km)", title="Tracer concentration")
slider = Slider(fig[4, 1:2], range=1:Nt, startvalue=1)
n = slider.value

u = @lift un($n)
b = @lift bn($n)
c = @lift cn($n)

hm = heatmap!(axu, y * 1e-3, z * 1e-3, u, colorrange=(min_u, max_u), colormap=:balance)
contour!(axu, y * 1e-3, z * 1e-3, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[2, 2], hm)

hm = heatmap!(axc, y * 1e-3, z * 1e-3, c, colorrange=(0, 0.5), colormap=:speed)
contour!(axc, y * 1e-3, z * 1e-3, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[3, 2], hm)

title_str = @lift "Baroclinic adjustment with GM at t = " * prettytime(times[$n])
ax_t = fig[1, 1:2] = Label(fig, title_str)

display(fig)

record(fig, filename * ".mp4", 1:Nt, framerate=8) do i
    @info "Plotting frame $i of $Nt"
    n[] = i
end