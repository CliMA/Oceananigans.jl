# # Tilted bottom boundary layer
#
# This example simulates a two-dimensional oceanic bottom boundary layer
# in a domain that's tilted with respect to gravity. We simulate the perturbation
# away from a constant along-slope (y-direction) velocity and a constant density stratification.
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
# We create a grid with finer resolution near the bottom,

using Oceananigans
using Oceananigans.Units
using Random

Random.seed!(42) # for reproducible results

Lx = 200meters
Lz = 100meters
Nx = 64
Nz = 64

refinement = 1.8 # controls spacing near the bottom (higher means finer spaced)
stretching = 10  # controls rate of stretching away from the bottom

h(k) = (Nz + 1 - k) / Nz
ζ(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

z_faces(k) = - Lz * (ζ(k) * Σ(k) - 1)

grid = RectilinearGrid(topology = (Periodic, Flat, Bounded),
                       size = (Nx, Nz),
                       x = (0, Lx),
                       z = z_faces)

# Let's make sure the grid spacing is both finer and near-uniform at the bottom,

using CairoMakie

scatterlines(zspacings(grid, Center()),
             axis = (ylabel = "Depth (m)",
                     xlabel = "Vertical spacing (m)"))

current_figure() #hide

# ## Tilting the domain
#
# We use a domain that's tilted with respect to gravity by

θ = 3 # degrees

# so that ``x`` is the across-slope direction, ``z`` is the slope-normal direction that
# is perpendicular to the bottom, and the unit vector anti-aligned with gravity is

ẑ = (sind(θ), 0, cosd(θ))

# Changing the vertical direction impacts both the `gravity_unit_vector`
# for `BuoyancyForce` as well as the `rotation_axis` for Coriolis forces,

buoyancy = BuoyancyForce(BuoyancyTracer(), gravity_unit_vector = .-ẑ)
coriolis = ConstantCartesianCoriolis(f = 1e-4, rotation_axis = ẑ)

# where above we used a constant Coriolis parameter ``f = 10^{-4} \, \rm{s}^{-1}``.
# The tilting also affects the kind of density stratified flows we can model.
# In particular, a constant density stratification in the tilted
# coordinate system

@inline constant_stratification(x, z, t, p) = p.N² * (x * p.ẑ[1] + z * p.ẑ[3])

# is _not_ periodic in ``x``. Thus we cannot explicitly model a constant stratification
# on an ``x``-periodic grid such as the one used here. Instead, we simulate periodic
# _perturbations_ away from the constant density stratification by imposing
# a constant stratification as a `BackgroundField`,

N² = 1e-5 # s⁻²
B∞_field = BackgroundField(constant_stratification, parameters=(; ẑ, N²))

# We choose to impose a bottom boundary condition of zero *total* diffusive buoyancy
# flux across the seafloor,
# ```math
# ∂_z B = ∂_z b + N^{2} \cos{\theta} = 0.
# ```
# This shows that to impose a no-flux boundary condition on the total buoyancy field ``B``,
# we must apply a boundary condition to the perturbation buoyancy ``b``,
# ```math
# ∂_z b = - N^{2} \cos{\theta}.
# ```

b_bcs = FieldBoundaryConditions(bottom = GradientBoundaryCondition(-N² * cosd(θ)))

# ## Bottom drag and along-slope interior velocity
#
# We impose bottom drag that follows Monin–Obukhov theory.
# We use `BulkDrag` to create the drag boundary conditions, which computes a
# quadratic drag proportional to the total velocity (including the background velocity):

V∞ = 0.1 # m s⁻¹
ℓ = 0.1  # roughness length (m)
ϰ = 0.4  # von Kármán constant

z₁ = first(znodes(grid, Center())) # height of the grid center closest to the bottom
cᴰ = (ϰ / log(z₁ / ℓ))^2

drag_bc = BulkDrag(coefficient=cᴰ, background_velocities=(0, V∞, 0))

u_bcs = FieldBoundaryConditions(bottom=drag_bc)
v_bcs = FieldBoundaryConditions(bottom=drag_bc)

# Note that, similar to the buoyancy boundary conditions, we had to
# include the background flow in the drag calculation.
#
# Let us also create a `BackgroundField` for the along-slope interior velocity:

V∞_field = BackgroundField(V∞)

# ## Create the `NonhydrostaticModel`
#
# We are now ready to create the model. We create a `NonhydrostaticModel` with a
# fifth-order `UpwindBiased` advection scheme and a constant viscosity and diffusivity.
# Here we use a smallish value of ``10^{-4} \, \rm{m}^2\, \rm{s}^{-1}``.

closure = ScalarDiffusivity(ν=1e-4, κ=1e-4)

model = NonhydrostaticModel(grid; buoyancy, coriolis, closure,
                            advection = UpwindBiased(order=5),
                            tracers = :b,
                            boundary_conditions = (u=u_bcs, v=v_bcs, b=b_bcs),
                            background_fields = (; b=B∞_field, v=V∞_field))

# Let's introduce a bit of random noise at the bottom of the domain to speed up the onset of
# turbulence:

noise(x, z) = 1e-3 * randn() * exp(-(10z)^2 / grid.Lz^2)
set!(model, u=noise, w=noise)

# ## Create and run a simulation
#
# We are now ready to create the simulation. We begin by setting the initial time step
# conservatively, based on the smallest horizontal grid spacing and the interior velocity.

Δt₀ = 0.5 * minimum_xspacing(grid) / V∞
simulation = Simulation(model, Δt = Δt₀, stop_time = 1day)

# We use a `TimeStepWizard` to adapt our time-step,

conjure_time_step_wizard!(simulation, IterationInterval(4), max_change=1.1, cfl=0.7)

# and also we add another callback to print a progress message,

using Printf

progress_message(sim) =
    @printf("Iteration: %04d, time: %s, Δt: %s, max|w|: %.1e m s⁻¹, wall time: %s\n",
            iteration(sim), prettytime(sim), prettytime(sim.Δt),
            maximum(abs, sim.model.velocities.w), prettytime(sim.run_wall_time))

add_callback!(simulation, progress_message, IterationInterval(200))

# ## Add outputs to the simulation
#
# We output the total buoyancy ``B``, the total along-slope velocity ``V``, and the
# ``y``-component of vorticity ``ω_y = ∂_z u - ∂_x w`` using the `NetCDFWriter`,
# which needs `NCDatasets` to be loaded:

u, v, w = model.velocities
b = model.tracers.b
B∞ = model.background_fields.tracers.b

B = b + B∞
V = v + V∞
ωy = ∂z(u) - ∂x(w)

outputs = (; u, V, w, B, ωy)

using NCDatasets

filename = joinpath(@__DIR__, "tilted_bottom_boundary_layer.nc")

simulation.output_writers[:fields] = NetCDFWriter(model, outputs; filename,
                                                  schedule = TimeInterval(20minutes),
                                                  overwrite_files = true)

# Now we just run it!

## Fail the docs build if this simulation produces NaNs #hide
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation) #hide
run!(simulation)

# ## Visualize the results
#
# We load the output as `FieldTimeSeries` and create an animation showing the
# ``y``-component of vorticity and the along-slope velocity, with buoyancy contours overlaid.

fig = Figure(size = (800, 600))

axis_kwargs = (xlabel = "Across-slope distance (m)",
               ylabel = "Slope-normal\ndistance (m)",
               limits = ((0, Lx), (0, Lz)))

ax_ω = Axis(fig[2, 1]; title = "Along-slope vorticity", axis_kwargs...)
ax_v = Axis(fig[3, 1]; title = "Along-slope velocity (v)", axis_kwargs...)

n = Observable(1)

ωyt = FieldTimeSeries(filename, "ωy")
Bt  = FieldTimeSeries(filename, "B")
Vt  = FieldTimeSeries(filename, "V")

ωyn = @lift ωyt[$n]
Bn  = @lift Bt[$n]
Vn  = @lift Vt[$n]

buoyancy_levels = -1e-3:5e-5:1e-3

hm_ω = heatmap!(ax_ω, ωyn, colorrange = (-0.015, +0.015), colormap = :balance)
Colorbar(fig[2, 2], hm_ω; label = "s⁻¹")
contour!(ax_ω, Bn, levels=buoyancy_levels, color=:black)

hm_v = heatmap!(ax_v, Vn, colorrange = (-V∞, +V∞), colormap = :balance)
Colorbar(fig[3, 2], hm_v; label = "m s⁻¹")
contour!(ax_v, Bn, levels=buoyancy_levels, color=:black)

times = ωyt.times
title = @lift "t = " * prettytime(times[$n])
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

current_figure() #hide
fig

# Finally, we record a movie.

frames = 1:length(times)

record(fig, "tilted_bottom_boundary_layer.mp4", frames, framerate=12) do i
    n[] = i
end
nothing #hide

# ![](tilted_bottom_boundary_layer.mp4)
