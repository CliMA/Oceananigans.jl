# # Tilted bottom boundary layer example
#
# This example simulates a two-dimensional oceanic bottom boundary layer
# in a domain that's tilted with respect to gravity. We simulate the perturbation
# away from a constant along-slope (y-direction) velocity constant density stratification.
# This perturbation develops into a turbulent bottom boundary layer due to momentum
# loss at the bottom boundary modeled with a quadratic drag law.
# 
# This example illustrates
#
#   * changing the direction of gravitational acceleration in the buoyancy model;
#   * changing the axis of rotation for Coriolis forces.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, NCDatasets, CairoMakie"
# ```
#
# ## The domain
#
# We create a ``400 × 100`` meter ``x, z`` grid with ``128 × 32`` cells
# and finer resolution near the bottom,

using Oceananigans

Lx = 400 # m
Lz = 100 # m
Nx = 128
Nz = 64

## Creates a grid with near-constant spacing `refinement * Lz / Nz`
## near the bottom:
refinement = 1.8 # controls spacing near surface (higher means finer spaced)
stretching = 10  # controls rate of stretching at bottom 

## "Warped" height coordinate
h(k) = (Nz + 1 - k) / Nz

## Linear near-surface generator
ζ(k) = 1 + (h(k) - 1) / refinement

## Bottom-intensified stretching function 
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

## Generating function
z_faces(k) = - Lz * (ζ(k) * Σ(k) - 1)

grid = RectilinearGrid(topology = (Periodic, Flat, Bounded),
                       size = (Nx, Nz),
                       x = (0, Lx),
                       z = z_faces,
                       halo = (3, 3))

# Let's make sure the grid spacing is both finer and near-uniform at the bottom,

using CairoMakie

lines(grid.Δzᵃᵃᶜ[1:Nz], grid.zᵃᵃᶜ[1:Nz],
      axis = (ylabel = "Depth (m)",
              xlabel = "Vertical spacing (m)"))

scatter!(grid.Δzᵃᵃᶜ[1:Nz], grid.zᵃᵃᶜ[1:Nz])

current_figure() # hide

# ## Tilting the domain
#
# We use a domain that's tilted with respect to gravity by

θ = 3 # degrees

# so that ``x`` is the along-slope direction, ``z`` is the across-sloce direction that
# is perpendicular to the bottom, and the unit vector anti-aligned with gravity is

ĝ = (sind(θ), 0, cosd(θ))

# Changing the vertical direction impacts both the `gravity_unit_vector`
# for `Buoyancy` as well as the `rotation_axis` for Coriolis forces,

buoyancy = Buoyancy(model = BuoyancyTracer(), gravity_unit_vector = ĝ)
coriolis = ConstantCartesianCoriolis(f = 1e-4, rotation_axis = ĝ)

# where we have used a constant Coriolis parameter ``f = 10⁻⁴ \rm{s}⁻¹``.
# The tilting also affects the kind of density stratified flows we can model.
# In particular, a constant density stratification in the tilted
# coordinate system

@inline constant_stratification(x, y, z, t, p) = p.N² * (x * p.ĝ[1] + z * p.ĝ[3])

# is _not_ periodic in ``x``. Thus we cannot explicitly model a constant stratification
# on an ``x``-periodic grid such as the one used here. Instead, we simulate periodic
# _perturbations_ away from the constant density stratification by imposing
# a constant stratification as a `BackgroundField`,

B_field = BackgroundField(constant_stratification, parameters=(; ĝ, N² = 1e-5))

# where ``N² = 10⁻⁵ \rm{s}⁻¹`` is the background buoyancy gradient.

# ## Bottom drag
#
# We impose bottom drag that follows Monin-Obukhov theory.
# We include the background flow in the drag calculation,
# which is the only effect the background flow enters the problem,

V∞ = 0.1 # m s⁻¹
z₀ = 0.1 # m (roughness length)
κ = 0.4 # von Karman constant
z₁ = znodes(Center, grid)[1] # Closest grid center to the bottom
cᴰ = (κ / log(z₁ / z₀))^2 # Drag coefficient

@inline drag_u(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + (v + p.V∞)^2) * u
@inline drag_v(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + (v + p.V∞)^2) * (v + p.V∞)

drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ, V∞))
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ, V∞))

u_bcs = FieldBoundaryConditions(bottom = drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom = drag_bc_v)

# ## Create the `NonhydrostaticModel`
#
# We are now ready to create the model. We create a `NonhydrostaticModel` with an
# `UpwindBiasedFifthOrder` advection scheme, a `RungeKutta3` timestepper,
# and a constant viscosity and diffusivity.

ν = 1e-4 # m² s⁻¹, small-ish
κ = ν
closure = ScalarDiffusivity(; ν, κ)

model = NonhydrostaticModel(; grid, buoyancy, coriolis, closure,
                            timestepper = :RungeKutta3,
                            advection = UpwindBiasedFifthOrder(),
                            tracers = :b,
                            boundary_conditions = (u=u_bcs, v=v_bcs),
                            background_fields = (; b=B_field))

# ## Create and run a simulation
#
# We are now ready to create the simulation. We begin by setting the initial time step
# conservatively, based on the smallest grid size of our domain and set-up a 

using Oceananigans.Units
using Oceananigans.Grids: min_Δz

simulation = Simulation(model, Δt = 0.5 * min_Δz(grid) / V∞, stop_time = 2days)

# We use `TimeStepWizard` to adapt our time-step and print a progress message,

using Printf

wizard = TimeStepWizard(max_change=1.1, cfl=0.7)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))

start_time = time_ns() # so we can print the total elapsed wall time

progress_message(sim) =
    @printf("Iteration: %04d, time: %s, Δt: %s, max|w|: %.1e m s⁻¹, wall time: %s\n",
            iteration(sim), prettytime(time(sim)),
            prettytime(sim.Δt), maximum(abs, sim.model.velocities.w),
            prettytime((time_ns() - start_time) * 1e-9))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(100))

# ## Add outputs to the simulation
#
# We add outputs to our model using the `NetCDFOutputWriter`,

u, v, w = model.velocities
b = model.tracers.b
B∞ = model.background_fields.tracers.b

B = b + B∞
V = v + V∞
ωy = ∂z(u) - ∂x(w)

outputs = (; u, V, w, B, ωy)

simulation.output_writers[:fields] = NetCDFOutputWriter(model, outputs;
                                                        filename = joinpath(@__DIR__, "tilted_bottom_boundary_layer.nc"),
                                                        schedule = TimeInterval(20minutes),
                                                        overwrite_existing = true)

# Now we just run it!

run!(simulation)

# ## Visualize the results
#
# First we load the required package to load NetCDF output files and define the coordinates for
# plotting using existing objects:

using NCDatasets, CairoMakie

xω, yω, zω = nodes(ωy)
xv, yv, zv = nodes(V)

# Read in the simulation's `output_writer` for the two-dimensional fields and then create an
# animation showing the ``y``-component of vorticity.

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

fig = Figure(resolution = (800, 440))
update_theme!(fontsize = 15)

axis_kwargs = (xlabel = "Across-slope distance (x)",
               ylabel = "Slope-normal\ndistance (z)",
               limits = ((0, Lx), (0, Lz)))

ax_ω = Axis(fig[2, 1]; title = "Along-slope vorticity", axis_kwargs...)
ax_v = Axis(fig[3, 1]; title = "Along-slope velocity (v)", axis_kwargs...)

n = Observable(1)

ωy = @lift ds["ωy"][:, 1, :, $n]
hm_ω = heatmap!(ax_ω, xω, zω, ωy, colorrange = (-0.015, +0.015), colormap = :balance)
Colorbar(fig[2, 2], hm_ω; label = "m s⁻¹")

V = @lift ds["V"][:, 1, :, $n]
V_max = @lift maximum(abs, ds["V"][:, 1, :, $n])

hm_v = heatmap!(ax_v, xv, zv, V, colorrange = (-V∞, +V∞), colormap = :balance)
Colorbar(fig[3, 2], hm_v; label = "m s⁻¹")

times = collect(ds["time"])
title = @lift "t = " * string(prettytime(times[$n]))
fig[1, :] = Label(fig, title, textsize=20, tellwidth=false)

# Finally, we record a movie.

frames = 1:length(times)

record(fig, "tilted_bottom_boundary_layer.mp4", frames, framerate=12) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    if i%5 == 0 print(msg * " \r") end
    n[] = i
end
nothing #hide

# ![](tilted_bottom_boundary_layer.mp4)

# Don't forget to close the NetCDF file!

close(ds)
