using Printf
using Statistics
using Random

using Oceananigans
using Oceananigans.Units
using Oceananigans: fields
using Oceananigans.TurbulenceClosures
using Oceananigans.TurbulenceClosures: FluxTapering

filename = "zonally_averaged_baroclinic_adjustment_withGM"

# Architecture
architecture = CPU()

# Domain
Ly = 1000kilometers  # north-south extent [m]
Lz = 1kilometers     # depth [m]

Ny = 64
Nz = 24

save_fields_interval = 0.5day
stop_time = 60days
Δt₀ = 5minutes

# We choose a regular grid though because of numerical issues that yet need to be resolved
grid = RectilinearGrid(architecture;
                       topology = (Flat, Bounded, Bounded), 
                       size = (Ny, Nz), 
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0),
                       halo = (3, 3))

coriolis = BetaPlane(latitude = -45)

κh = 1e4  # [m² s⁻¹] horizontal diffusivity
νh = 1e4  # [m² s⁻¹] horizontal viscosity
κz = 1e-2 # [m² s⁻¹] vertical diffusivity
νz = 1e-2 # [m² s⁻¹] vertical viscosity

horizontal_diffusivity = HorizontalScalarDiffusivity(κ=100)
vertical_diffusivity = VerticalScalarDiffusivity(κ=1e-2)
convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=1)

gerdes_koberle_willebrand_tapering = FluxTapering(1e-2)
gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = 1000,
                                                                κ_symmetric = 900,
                                                                slope_limiter = gerdes_koberle_willebrand_tapering)
#####
##### Model building
#####

@info "Building a model..."

closures = (vertical_diffusivity, horizontal_diffusivity, convective_adjustment, gent_mcwilliams_diffusivity)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    coriolis = coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    closure = closures,
                                    tracers = (:b, :c),
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    free_surface = ImplicitFreeSurface())

@info "Built $model."

#####
##### Initial conditions
#####

"""
Linear ramp from 0 to 1 between -Δy/2 and +Δy/2.

For example:

y < y₀           => ramp = 0
y₀ < y < y₀ + Δy => ramp = y / Δy
y > y₀ + Δy      => ramp = 1
"""
ramp(y, Δy) = min(max(0, y/Δy + 1/2), 1)

# Parameters
N² = 4e-6 # [s⁻²] buoyancy frequency / stratification
M² = 8e-8 # [s⁻²] horizontal buoyancy gradient

Δy = 100kilometers
Δz = 100

Δc = 2Δy
Δb = Δy * M²
ϵb = 1e-2 * Δb # noise amplitude

bᵢ(x, y, z) = N² * z + Δb * ramp(y, Δy)
cᵢ(x, y, z) = exp(-y^2 / 2Δc^2) * exp(-(z + Lz/4)^2 / 2Δz^2)

set!(model, b=bᵢ, c=cᵢ)

#####
##### Simulation building
#####

simulation = Simulation(model, Δt=Δt₀, stop_time=stop_time)

# add timestep wizard callback
wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=10minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# add progress callback
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

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

#####
##### Output
#####

Redi_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = (b=0, c=0),
                                                     κ_symmetric = (b=1, c=0),
                                                     slope_limiter = gerdes_koberle_willebrand_tapering)

#=
dependencies = (Redi_diffusivity,
                model.diffusivity_fields,
                Val(1),
                model.velocities,
                model.tracers,
                model.clock,
                model.buoyancy)

using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ

∇_q_op = KernelFunctionOperation{Center, Center, Center}(∇_dot_qᶜ,
                                                         grid,
                                                         computed_dependencies = dependencies)

# R(b) eg the Redi operator applied to buoyancy
Rb = Field(∇_q_op)

outputs = merge(fields(model), (; Rb))
=#

outputs = fields(model)

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(save_fields_interval),
                                                      prefix = filename * "_fields",
                                                      overwrite_existing = true)

@info "Running the simulation..."

run!(simulation, pickup=false)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)

#####
##### Visualize
#####

using GLMakie

fig = Figure(resolution = (1400, 700))

filepath = filename * "_fields.jld2"

ut = FieldTimeSeries(filepath, "u")
bt = FieldTimeSeries(filepath, "b")
ct = FieldTimeSeries(filepath, "c")
#rt = FieldTimeSeries(filepath, "Rb")

# Build coordinates, rescaling the vertical coordinate
x, y, z = nodes((Center, Center, Center), grid)

zscale = 1
z = z .* zscale

#####
##### Plot buoyancy...
#####

times = bt.times
Nt = length(times)

un(n) = interior(ut[n])[1, :, :]
bn(n) = interior(bt[n])[1, :, :]
cn(n) = interior(ct[n])[1, :, :]

#rn(n) = interior(rt[n])[1, :, :]
#@show max_r = maximum(abs, rn(Nt))
#@show min_r = - max_r

@show min_c = 0
@show max_c = 1
@show max_u = maximum(abs, un(Nt))
min_u = - max_u

slider = Slider(fig[3, 1], range=1:Nt, startvalue=1)
n = slider.value

u = @lift interior(ut[$n], 1, :, :)
b = @lift interior(bt[$n], 1, :, :)
c = @lift interior(ct[$n], 1, :, :)
#r = @lift rn($n)

ax = Axis(fig[1, 1], title="Zonal velocity")
hm = heatmap!(ax, y * 1e-3, z * 1e-3, u, colorrange=(min_u, max_u), colormap=:balance)
contour!(ax, y * 1e-3, z * 1e-3, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[1, 2], hm)

ax = Axis(fig[2, 1], title="Tracer concentration")
hm = heatmap!(ax, y * 1e-3, z * 1e-3, c, colorrange=(0, 0.5), colormap=:speed)
contour!(ax, y * 1e-3, z * 1e-3, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[2, 2], hm)

#=
ax = Axis(fig[3, 1], title="R(b)")
hm = heatmap!(ax, y * 1e-3, z * 1e-3, r, colorrange=(min_r, max_r), colormap=:balance)
contour!(ax, y * 1e-3, z * 1e-3, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[3, 2], hm)
=#

title_str = @lift "Baroclinic adjustment with GM at t = " * prettytime(times[$n])
ax_t = fig[0, :] = Label(fig, title_str)

display(fig)

record(fig, filename * ".mp4", 1:Nt, framerate=8) do i
    @info "Plotting frame $i of $Nt"
    n[] = i
end

