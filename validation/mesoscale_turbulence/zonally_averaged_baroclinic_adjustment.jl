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

Ny = 128
Nz = 40

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

Δy, Δz = Ly/Ny, Lz/Nz

𝒜 = Δz/Δy   # Grid cell aspect ratio.

κh = 0.1    # [m² s⁻¹] horizontal diffusivity
νh = 0.1    # [m² s⁻¹] horizontal viscosity
κz = 𝒜 * κh # [m² s⁻¹] vertical diffusivity
νz = 𝒜 * νh # [m² s⁻¹] vertical viscosity

vertical_closure = VerticalScalarDiffusivity(ν = νz, κ = κz)
horizontal_closure = HorizontalScalarDiffusivity(ν = νh, κ = κh)
diffusive_closures = (vertical_closure, horizontal_closure)

gerdes_koberle_willebrand_tapering = FluxTapering(1e-2)
gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = 1000,
                                                                κ_symmetric = 900,
                                                                slope_limiter = gerdes_koberle_willebrand_tapering)
#####
##### Model building
#####

@info "Building a model..."

closures = (diffusive_closures..., gent_mcwilliams_diffusivity)

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
wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=20minutes)
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

#####
##### Output
#####

Redi_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = (b=0, c=0),
                                                     κ_symmetric = (b=1, c=0),
                                                     slope_limiter = gerdes_koberle_willebrand_tapering)

dependencies = (Redi_diffusivity,
                model.tracers.b,
                Val(1),
                model.clock,
                model.diffusivity_fields,
                model.tracers,
                model.buoyancy,
                model.velocities)

using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ

∇_q_op = KernelFunctionOperation{Center, Center, Center}(∇_dot_qᶜ,
                                                         grid,
                                                         computed_dependencies = dependencies)

# R(b) eg the Redi operator applied to buoyancy
Rb = Field(∇_q_op)

outputs = merge(fields(model), (; Rb))

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(save_fields_interval),
                                                      prefix = filename * "_fields",
                                                      force = true)

@info "Running the simulation..."

run!(simulation, pickup=false)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)

#=

#####
##### Visualize
#####

using GLMakie

fig = Figure(resolution = (1400, 700))

filepath = filename * "_fields.jld2"

ut = FieldTimeSeries(filepath, "u")
bt = FieldTimeSeries(filepath, "b")
ct = FieldTimeSeries(filepath, "c")
rt = FieldTimeSeries(filepath, "Rb")

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
rn(n) = interior(rt[n])[1, :, :]

@show min_c = 0
@show max_c = 1
@show max_u = maximum(abs, un(Nt))
min_u = - max_u

@show max_r = maximum(abs, rn(Nt))
@show min_r = - max_r

n = Node(1)
u = @lift un($n)
b = @lift bn($n)
c = @lift cn($n)
r = @lift rn($n)

ax = Axis(fig[1, 1], title="Zonal velocity")
hm = heatmap!(ax, y * 1e-3, z * 1e-3, u, colorrange=(min_u, max_u), colormap=:balance)
contour!(ax, y * 1e-3, z * 1e-3, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[1, 2], hm)

ax = Axis(fig[2, 1], title="Tracer concentration")
hm = heatmap!(ax, y * 1e-3, z * 1e-3, c, colorrange=(0, 0.5), colormap=:speed)
contour!(ax, y * 1e-3, z * 1e-3, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[2, 2], hm)

ax = Axis(fig[3, 1], title="R(b)")
hm = heatmap!(ax, y * 1e-3, z * 1e-3, r, colorrange=(min_r, max_r), colormap=:balance)
contour!(ax, y * 1e-3, z * 1e-3, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[3, 2], hm)

title_str = @lift "Baroclinic adjustment with GM at t = " * prettytime(times[$n])
ax_t = fig[0, :] = Label(fig, title_str)

display(fig)

record(fig, filename * ".mp4", 1:Nt, framerate=8) do i
    @info "Plotting frame $i of $Nt"
    n[] = i
end

=#
