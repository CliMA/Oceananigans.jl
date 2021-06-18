# using Pkg
# pkg"add Oceananigans GLMakie"

using Printf
using Statistics
using GLMakie
using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries

# # Vertically-stretched grid
#
# We build a vertically stretched grid...

Nx = 32
Ny = 64
Nz = 32

const Lx = 1000kilometers # channel east-west width [m]
const Ly = 2000kilometers # channel north-south width [m]
const Lz = 3kilometers    # channel depth [m]

#=
## Stretching function
z_faces(k) = - Lz * (1 - ((k - 1) / Nz)^(3/4))

grid = VerticallyStretchedRectilinearGrid(topology = (Periodic, Bounded, Bounded),
                                          size = (Nx, Ny, Nz),
                                          halo = (3, 3, 3),
                                          x = (0, Lx),
                                          y = (0, Ly),
                                          z_faces = z_faces)

fig = Figure(resolution=(200, 600))
ax = Axis(fig[1, 1])
scatter!(ax, grid.Δzᵃᵃᶜ[1:Nz], grid.zᵃᵃᶜ[1:Nz])
=#

grid = RegularRectilinearGrid(topology = (Periodic, Bounded, Bounded),
                              size = (Nx, Ny, Nz),
                              halo = (3, 3, 3),
                              x = (0, Lx),
                              y = (0, Ly),
                              z = (-Lz, 0))

## Visualize grid
@show grid

# # Boundary conditions
#
# A channel-centered jet and overturning circulation are driven by wind stress
# and an alternating pattern of surface cooling and surface heating with
# parameters

Qᵇ = 0 # 1e-9            # buoyancy flux magnitude [m² s⁻³]
y_shutoff = 5/6 * Ly # shutoff location for buoyancy flux [m]
τ = 0 # 1e-4             # surface kinematic wind stress [m² s⁻²]
μ = 1 / 100days      # bottom drag damping time-scale [s⁻¹]

# The buoyancy flux has a sinusoidal pattern in `y`,

@inline buoyancy_flux(x, y, t, p) = ifelse(y < p.y_shutoff, p.Qᵇ * cos(3π * y / p.Ly), 0)

buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, parameters=(Ly=grid.Ly, y_shutoff=y_shutoff, Qᵇ=Qᵇ))

# At the surface we impose a wind stress with sinusoidal variation in `y`,

@inline u_stress(x, y, t, p) = - p.τ * sin(π * y / p.Ly)

u_stress_bc = FluxBoundaryCondition(u_stress, parameters=(τ=τ, Ly=grid.Ly))

# Linear bottom drag on `u` and `v` provides a sink of momentum

@inline u_drag(x, y, t, u, μ) = - μ * u
@inline v_drag(x, y, t, v, μ) = - μ * v

u_drag_bc = FluxBoundaryCondition(u_drag, field_dependencies=:u, parameters=μ)
v_drag_bc = FluxBoundaryCondition(v_drag, field_dependencies=:v, parameters=μ)

# To summarize,

b_bcs = TracerBoundaryConditions(grid, top = buoyancy_flux_bc)
u_bcs = UVelocityBoundaryConditions(grid, top = u_stress_bc, bottom = u_drag_bc)
v_bcs = VVelocityBoundaryConditions(grid, bottom = v_drag_bc)

# # Sponge layer
#
# A forcing term that relaxes the buoyancy field to a prescribed stratification
# at the northern wall produces an overturning circulation.
#
# We declare parameters as `const` so we can reference them as global variables
# in our forcing functions.

const Δb = 0.01                 # cross-channel buoyancy jump [m s⁻²]
const N² = 1e-5                 # cross-channel buoyancy jump [m s⁻²]
const h = 1kilometer            # decay scale of stable stratification (N² ≈ Δb / h) [m]
const y_sponge = 1900kilometers # southern boundary of sponge layer [m]

## Target (and initial) buoyancy profile
@inline b_target(x, y, z, t) = Δb * y / Ly + N² * h * exp(z / h)

## Mask that limits sponge layer to a thin region near the northern boundary
@inline northern_mask(x, y, z) = max(0, y - y_sponge) / (Ly - y_sponge)

b_forcing = Relaxation(target=b_target, mask=northern_mask, rate=1/7days)

# # Turbulence closures
#
# A horizontally Laplacian diffusivity destroys enstrophy and buoyancy variance
# created by mesoscale turbulence, while a convective adjustment scheme creates
# a surface mixed layer due to surface cooling.

horizontal_diffusivity = AnisotropicDiffusivity(νh=10)

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
                                                                convective_νz = 1.0,
                                                                background_κz = 1e-4,
                                                                background_νz = 1e-4)

# # Model building
#
# We build a model on a BetaPlane with an ImplicitFreeSurface.

model = HydrostaticFreeSurfaceModel(architecture = CPU(),                                           
                                    grid = grid,
                                    free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1),
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = BetaPlane(latitude=-45),
                                    closure = (convective_adjustment, horizontal_diffusivity),
                                    tracers = :b,
                                    boundary_conditions = (b=b_bcs, u=u_bcs, v=v_bcs),
                                    forcing = (b=b_forcing,),
                                    )

# # Initial Conditions
#
# Our initial condition is an unstable, geostrophically-balanced shear flow
# and stable buoyancy stratification superposed with surface-concentrated
# random noise.
#
# The geostrophic streamfunction is
#
# ```math
# ψ(y, z) = Δb (z + L_z) y / (f L_y) \, ,
# ```
#
# where ``f(y) = f₀ + β y``. The geostrophic buoyancy field is then
#
# ```math
# b(y) = f ∂_z ψ = Δb y / L_y \, ,
# ```
#
# consistent with the ``y``-dependent part of `b_target`, and the
# geostrophic zonal velocity is
#
# ```math
# u = - ∂_y ψ = - Δb (z + L_z) / (f L_y) \, .
# ```
#
# Recall that our austral focus implies that f < 0.
#
# We scale the initial noise with the friction velocity implied by wind stress,
# and concentrate the noise in the upper tenth of the domain.

## Random noise concentrated at the top
u★ = 1e-3 * sqrt(τ)
h★ = Lz / 10
ϵ(z) = u★ * exp(z / h★)

f(y, c::BetaPlane) = c.f₀ + c.β * y
f(y, c::FPlane) = c.f

uᵢ(x, y, z) = - Δb * (z + Lz) / (f(y, model.coriolis) * Ly)
vᵢ(x, y, z) = 0
bᵢ(x, y, z) = b_target(x, y, z, 0)

set!(model, u=uᵢ, v=vᵢ, b=bᵢ)

# # Simulation setup
#
# We set up a simulation with adaptive time-stepping and a simple progress message.

using Oceananigans.Diagnostics: accurate_cell_advection_timescale

wizard = TimeStepWizard(cfl=0.2, Δt=1minutes, max_change=1.1, max_Δt=10minutes, min_Δt=1minute,
                        cell_advection_timescale = accurate_cell_advection_timescale)

print_progress(sim) = @printf("[%05.2f%%] i: %d, t: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
                              100 * (sim.model.clock.time / sim.stop_time),
                              sim.model.clock.iteration,
                              prettytime(sim.model.clock.time),
                              maximum(abs, sim.model.velocities.u),
                              maximum(abs, sim.model.velocities.v),
                              maximum(abs, sim.model.velocities.w),
                              prettytime(sim.Δt.Δt))

simulation = Simulation(model, Δt=wizard, stop_time=1day, progress=print_progress, iteration_interval=1)

u, v, w = model.velocities
b = model.tracers.b

ζ = ComputedField(∂x(v) - ∂y(u))

B = AveragedField(b, dims=(1, 2))
χ_op = @at (Center, Center, Center) ∂x(b - B)^2 + ∂y(b - B)^2 + ∂z(b - B)^2
χ = ComputedField(χ_op)

outputs = merge(model.velocities, model.tracers, (ζ=ζ, χ=χ))

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(100days),
                                                        prefix = "eddying_channel",
                                                        force = true)

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(10minutes),
                                                      prefix = "eddying_channel",
                                                      field_slicer = nothing,
                                                      force = true)

try
    run!(simulation, pickup=false)
catch err
    showerror(stdout, err)
end

# # Visualizing the solution with GLMakie
#
# We make a volume rendering of the solution using GLMakie.

u_timeseries = FieldTimeSeries("eddying_channel.jld2", "u")
v_timeseries = FieldTimeSeries("eddying_channel.jld2", "v")
w_timeseries = FieldTimeSeries("eddying_channel.jld2", "w")
b_timeseries = FieldTimeSeries("eddying_channel.jld2", "b")
ζ_timeseries = FieldTimeSeries("eddying_channel.jld2", "ζ")
χ_timeseries = FieldTimeSeries("eddying_channel.jld2", "χ")

xζ, yζ, zζ = nodes((Face, Face, Center), grid)
xu, yu, zu = nodes((Face, Center, Center), grid)
xv, yv, zv = nodes((Center, Face, Center), grid)
xw, yw, zw = nodes((Center, Center, Face), grid)
xc, yc, zc = nodes((Center, Center, Center), grid)

kwargs = (algorithm=:absorption, absorption=0.5, colormap=:balance, show_axis=true)

iter = Node(1)

u′ = @lift interior(u_timeseries[$iter])
v′ = @lift interior(v_timeseries[$iter])
w′ = @lift interior(w_timeseries[$iter])
b′ = @lift interior(b_timeseries[$iter]) .- mean(interior(b_timeseries[$iter]))
ζ′ = @lift interior(ζ_timeseries[$iter])
χ′ = @lift interior(χ_timeseries[$iter])

function symmetric_lims(q, iter)
    q_max = maximum(abs, interior(q[iter]))
    return (-q_max, q_max)
end

u_lims = @lift symmetric_lims(u_timeseries, $iter)
v_lims = @lift symmetric_lims(v_timeseries, $iter)
w_lims = @lift symmetric_lims(w_timeseries, $iter)
b_lims = @lift symmetric_lims(b_timeseries, $iter)
ζ_lims = @lift symmetric_lims(ζ_timeseries, $iter)
χ_lims = @lift symmetric_lims(χ_timeseries, $iter)

fig = Figure(resolution = (2000, 1600))

ax_u = fig[1, 1] = LScene(fig)
ax_v = fig[1, 2] = LScene(fig)
ax_w = fig[2, 1] = LScene(fig)
ax_b = fig[2, 2] = LScene(fig)

volume!(ax_u, xu * 1e-3, yu * 1e-3, zu / 10, u′; colorrange=u_lims, kwargs...) 
volume!(ax_v, xv * 1e-3, yv * 1e-3, zu / 10, v′; colorrange=u_lims, kwargs...) 
volume!(ax_w, xw * 1e-3, yw * 1e-3, zu / 10, w′; colorrange=w_lims, kwargs...) 
volume!(ax_b, xc * 1e-3, yc * 1e-3, zc / 10, b′; colorrange=b_lims, kwargs...) 

#volume!(ax_ζ, xζ * 1e-3, yζ * 1e-3, zζ / 10, ζ′; colorrange=ζ_lims, kwargs...) 
#volume!(ax_χ, xc * 1e-3, yc * 1e-3, zc / 10, χ′; colorrange=χ_lims, kwargs...) 

nframes = length(ζ_timeseries.times)

record(fig, "eddying_channel.mp4", 1:nframes, framerate=12) do i
    @info "Plotting frame $i of $nframes..."
    iter[] = i
end
