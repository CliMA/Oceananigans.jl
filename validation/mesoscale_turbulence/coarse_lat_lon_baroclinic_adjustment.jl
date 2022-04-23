using Printf
using Statistics
using Random
using Oceananigans
using Oceananigans.Units
using GLMakie

gradient = "φ"
filename = "coarse_baroclinic_adjustment_" * gradient

# Architecture
architecture = CPU()

# Domain
Lz = 1kilometers     # depth [m]
Ny = 20
Nz = 20
save_fields_interval = 0.5day
stop_time = 30days
Δt = 20minutes

grid = LatitudeLongitudeGrid(architecture;
                             topology = (Bounded, Bounded, Bounded),
                             size = (Ny, Ny, Nz), 
                             longitude = (-5, 5),
                             latitude = (40, 50),
                             z = (-Lz, 0),
                             halo = (3, 3, 3))

coriolis = HydrostaticSphericalCoriolis()

Δy = 1000kilometers / Ny
@show κh = νh = Δy^4 / 10days
vertical_closure = VerticalScalarDiffusivity(ν=1e-2, κ=1e-4)
horizontal_closure = HorizontalScalarBiharmonicDiffusivity(ν=νh, κ=κh)

gerdes_koberle_willebrand_tapering = FluxTapering(1e-2)
gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3,
                                                                κ_symmetric=1e3,
                                                                slope_limiter=gerdes_koberle_willebrand_tapering)

closures = (vertical_closure, horizontal_closure, gent_mcwilliams_diffusivity)

@info "Building a model..."

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    coriolis = coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    closure = closures,
                                    tracers = (:b, :c),
                                    momentum_advection = VectorInvariant(),
                                    tracer_advection = WENO5(),
                                    free_surface = ImplicitFreeSurface())

@info "Built $model."

"""
Linear ramp from 0 to 1 between -Δy/2 and +Δy/2.

For example:

y < y₀           => ramp = 0
y₀ < y < y₀ + Δy => ramp = y / Δy
y > y₀ + Δy      => ramp = 1
"""
function ramp(λ, φ, Δ)
    gradient == "λ" && return min(max(0, λ / Δ + 1/2), 1)
    gradient == "φ" && return min(max(0, (φ - 45) / Δ + 1/2), 1)
end

# Parameters
N² = 4e-6 # [s⁻²] buoyancy frequency / stratification
M² = 8e-8 # [s⁻²] horizontal buoyancy gradient

Δφ = 1 # degree
Δz = 100

Δc = 100kilometers * 2Δφ
Δb = 100kilometers * Δφ * M²
ϵb = 1e-2 * Δb # noise amplitude

bᵢ(λ, φ, z) = N² * z + Δb * ramp(λ, φ, Δφ)
cᵢ(λ, φ, z) = exp(-φ^2 / 2Δc^2) * exp(-(z + Lz/4)^2 / 2Δz^2)

set!(model, b=bᵢ, c=cᵢ)

#####
##### Simulation building
#####

simulation = Simulation(model; Δt, stop_time)

# add timestep wizard callback
wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

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

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                      schedule = TimeInterval(save_fields_interval),
                                                      filename = filename * "_fields",
                                                      overwrite_existing = true)

@info "Running the simulation..."

run!(simulation)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)

#####
##### Visualize
#####

fig = Figure(resolution = (1400, 700))

filepath = filename * "_fields.jld2"

ut = FieldTimeSeries(filepath, "u")
bt = FieldTimeSeries(filepath, "b")
ct = FieldTimeSeries(filepath, "c")

# Build coordinates, rescaling the vertical coordinate
x, y, z = nodes((Center, Center, Center), grid)

#####
##### Plot buoyancy...
#####

times = bt.times
Nt = length(times)

if gradient == "φ" # average in x
    un(n) = interior(mean(ut[n], dims=1), 1, :, :)
    bn(n) = interior(mean(bt[n], dims=1), 1, :, :)
    cn(n) = interior(mean(ct[n], dims=1), 1, :, :)
else # average in y
    un(n) = interior(mean(ut[n], dims=2), :, 1, :)
    bn(n) = interior(mean(bt[n], dims=2), :, 1, :)
    cn(n) = interior(mean(ct[n], dims=2), :, 1, :)
end

@show min_c = 0
@show max_c = 1
@show max_u = maximum(abs, un(Nt))
min_u = - max_u

axu = Axis(fig[2, 1], xlabel="$gradient (deg)", ylabel="z (m)", title="Zonal velocity")
axc = Axis(fig[3, 1], xlabel="$gradient (deg)", ylabel="z (m)", title="Tracer concentration")
slider = Slider(fig[4, 1:2], range=1:Nt, startvalue=1)
n = slider.value

u = @lift un($n)
b = @lift bn($n)
c = @lift cn($n)

hm = heatmap!(axu, y, z, u, colorrange=(min_u, max_u), colormap=:balance)
contour!(axu, y, z, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[2, 2], hm)

hm = heatmap!(axc, y, z, c, colorrange=(0, 0.5), colormap=:speed)
contour!(axc, y, z, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[3, 2], hm)

title_str = @lift "Baroclinic adjustment with GM at t = " * prettytime(times[$n])
ax_t = fig[1, 1:2] = Label(fig, title_str)

display(fig)

record(fig, filename * ".mp4", 1:Nt, framerate=8) do i
    @info "Plotting frame $i of $Nt"
    n[] = i
end

