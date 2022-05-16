using Printf
using Statistics
using Random
using SpecialFunctions

using Oceananigans
using Oceananigans.Units
using Oceananigans.Fields: FunctionField
using Oceananigans: fields
using Oceananigans.TurbulenceClosures: FluxTapering, VerticallyImplicitTimeDiscretization

filename = "zonally_averaged_baroclinic_adjustment"

# Architecture
architecture = CPU()

# Domain
Ly = 1000kilometers  # north-south extent [m]
Lz = 1kilometers     # depth [m]

Ny = 20
Nz = 20

save_fields_interval = 0.5day
stop_time = 30days
Δt₀ = 1minutes

# Parameters
N² = 4e-6 # [s⁻²] buoyancy frequency / stratification
M² = 8e-8 # [s⁻²] horizontal buoyancy gradient

Δy = 100kilometers
Δz = 200

Δc = 2Δy
Δb = Δy * M²
ϵb = 1e-2 * Δb # noise amplitude

# We choose a regular grid though because of numerical issues that yet need to be resolved
grid = RectilinearGrid(architecture;
                       topology = (Flat, Bounded, Bounded), 
                       size = (Ny, Nz), 
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0),
                       halo = (3, 3))

coriolis = BetaPlane(latitude = -45)

νzz = 1e0
vertical_viscosity = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=νzz)

κh = νh = (Ly / Ny)^4 / 10days
horizontal_biharmonic_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=νh)

κᴳᴹ = 1000
horizontal_viscosity = HorizontalScalarDiffusivity(ν=κᴳᴹ)
gerdes_koberle_willebrand_tapering = FluxTapering(1e-2)
advective_gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = κᴳᴹ,
                                                                          skew_flux_scheme = WENO5(),
                                                                          slope_limiter = gerdes_koberle_willebrand_tapering)

redi_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_symmetric = 900,
                                                     slope_limiter = gerdes_koberle_willebrand_tapering)
#####
##### Model building
#####

@info "Building a model..."

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    coriolis = coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    #closure = (gent_mcwilliams_diffusivity, horizontal_viscosity),
                                    closure = advective_gent_mcwilliams_diffusivity,
                                    #closure = (vertical_viscosity, horizontal_biharmonic_viscosity),
                                    #closure = vertical_viscosity,
                                    tracers = (:b, :c),
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    free_surface = ImplicitFreeSurface())

@info "Built $model."

#####
##### Initial condition
#####

# ψ = Δb / f * (z + Lz) * tanh(y / δ)
# b = f ∂z ψ = Δb * tanh(y / δ)
# u = - ∂y ψ = - Δb * (z + Lz) / (f * δ * cosh(y / δ)^2)
f = model.coriolis.f₀
uᵢ(x, y, z) = - Δb * (z + Lz) / (f * Δy * cosh(y / Δy)^2)
bᵢ(x, y, z) = Δb * tanh(y / Δy) + N² * z
cᵢ(x, y, z) = exp(-y^2 / 2Δc^2) * exp(-(z + Lz/4)^2 / 2Δz^2)

set!(model, u=uᵢ, b=bᵢ, c=cᵢ)

#####
##### Simulation building
#####

simulation = Simulation(model, Δt=Δt₀, stop_time=stop_time)

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

u, v, w = model.velocities
ζz = ∂z(∂x(v) - ∂y(u))

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(fields(model), (; ζz)),
                                                      schedule = TimeInterval(save_fields_interval),
                                                      filename = filename * "_fields",
                                                      overwrite_existing = true)

@info "Running the simulation..."

run!(simulation)

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
ζzt = FieldTimeSeries(filepath, "ζz")

# Build coordinates, rescaling the vertical coordinate
x, y, z = nodes((Center, Center, Center), grid)

#####
##### Plot buoyancy...
#####

times = bt.times
Nt = length(times)

un(n) = interior(ut[n], 1, :, :)
bn(n) = interior(bt[n], 1, :, :)
cn(n) = interior(ct[n], 1, :, :)
ζzn(n) = interior(ct[n], 1, :, :)

@show min_c = 0
@show max_c = 1
@show max_u = maximum(abs, un(Nt)) / 2
min_u = - max_u

y *= 1e-3

axu = Axis(fig[2, 1], xlabel="y (km)", ylabel="z (m)", title="Zonal velocity")
axc = Axis(fig[3, 1], xlabel="y (km)", ylabel="z (m)", title="Vorticity vertical gradient")
slider = Slider(fig[4, 1:2], range=1:Nt, startvalue=1)
n = slider.value

u = @lift un($n)
b = @lift bn($n)
c = @lift cn($n)

hm = heatmap!(axu, y, z, u, colorrange=(min_u, max_u), colormap=:balance)
contour!(axu, y, z, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[2, 2], hm)

#hm = heatmap!(axc, y, z, ζz, colorrange=(0, 0.5), colormap=:speed)
hm = heatmap!(axc, y, z, c, colorrange=(0, 0.5), colormap=:speed)
contour!(axc, y, z, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[3, 2], hm)

title_str = @lift "Baroclinic adjustment at t = " * prettytime(times[$n])
ax_t = fig[1, 1:2] = Label(fig, title_str)

display(fig)

record(fig, filename * ".mp4", 1:Nt, framerate=8) do i
    @info "Plotting frame $i of $Nt"
    n[] = i
end

