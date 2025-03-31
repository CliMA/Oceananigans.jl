# # Tilted bottom boundary layer example
#
# This example is based on the similar [Oceananigans
# example](https://clima.github.io/OceananigansDocumentation/stable/literated/tilted_bottom_boundary_layer/)
# and simulates a two-dimensional oceanic bottom boundary layer in a domain that's tilted with
# respect to gravity. We simulate the perturbation away from a constant along-slope
# (y-direction) velocity constant density stratification.  This perturbation develops into a
# turbulent bottom boundary layer due to momentum loss at the bottom boundary.
# 
#
# First let's make sure we have all required packages installed.
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Oceanostics, Rasters, CairoMakie"
# ```
#
# ## Grid
#
# We start by creating a ``x, z`` grid with 64² cells and finer resolution near the bottom:

using Oceananigans
using Oceananigans.Units

Lx = 200meters
Lz = 100meters
Nx = 64
Nz = 64

refinement = 1.8 # controls spacing near surface (higher means finer spaced)
stretching = 10  # controls rate of stretching at bottom

h(k) = (Nz + 1 - k) / Nz
ζ(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))
z_faces(k) = - Lz * (ζ(k) * Σ(k) - 1)

grid = RectilinearGrid(topology = (Periodic, Flat, Bounded), size = (Nx, Nz),
                       x = (0, Lx), z = z_faces)

# Note that, with the `z` faces defined as above, the spacings near the bottom are approximately
# constant, becoming progressively coarser moving up.
#
# ## Tilting the domain
#
# We use a domain that's tilted with respect to gravity by

θ = 5; # degrees

# so that ``x`` is the along-slope direction, ``z`` is the across-sloce direction that
# is perpendicular to the bottom, and the unit vector anti-aligned with gravity is

ĝ = [sind(θ), 0, cosd(θ)]

# Changing the vertical direction impacts both the `gravity_unit_vector` for `Buoyancy` as well as
# the `rotation_axis` for Coriolis forces,

buoyancy = BuoyancyForce(BuoyancyTracer(), gravity_unit_vector = -ĝ)

f₀ = 1e-4/second
coriolis = ConstantCartesianCoriolis(f = f₀, rotation_axis = ĝ)

# The tilting also affects the kind of density stratified flows we can model. The simulate an
# environment that's uniformly stratified, with a stratification frequency

N² = 1e-5/second^2;

# In a tilted coordinate, this can be achieved with

@inline constant_stratification(x, z, t, p) = p.N² * (x * p.ĝ[1] + z * p.ĝ[3]);

# However, this distribution is _not_ periodic in ``x`` and can't be explicitly modelled on an
# ``x``-periodic grid such as the one used here. Instead, we simulate periodic _perturbations_ away
# from the constant density stratification by imposing a constant stratification as a
# `BackgroundField`,

B_field = BackgroundField(constant_stratification, parameters=(; ĝ, N²))

# ## Bottom drag
#
# We impose bottom drag that follows Monin-Obukhov theory and include the background flow in the
# drag calculation, which is the only effect the background flow has on the problem

V∞ = 0.1meters/second
z₀ = 0.1meters # (roughness length)
κ = 0.4 # von Karman constant
z₁ = znodes(grid, Center())[1] # Closest grid center to the bottom
cᴰ = (κ / log(z₁ / z₀))^2 # Drag coefficient

@inline drag_u(x, t, u, v, p) = - p.cᴰ * √(u^2 + (v + p.V∞)^2) * u
@inline drag_v(x, t, u, v, p) = - p.cᴰ * √(u^2 + (v + p.V∞)^2) * (v + p.V∞)

drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ, V∞))
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ, V∞))

u_bcs = FieldBoundaryConditions(bottom = drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom = drag_bc_v)

# ## Create model and simulation
#
# We are now ready to create the model. We create a `NonhydrostaticModel` with a 5th
# `UpwindBiased` advection scheme, a `RungeKutta3` timestepper, and a constant viscosity
# and diffusivity.

closure = ScalarDiffusivity(ν=2e-4, κ=2e-4)

model = NonhydrostaticModel(; grid, buoyancy, coriolis, closure,
                            timestepper = :RungeKutta3,
                            advection = UpwindBiased(order=5),
                            tracers = :b,
                            boundary_conditions = (u=u_bcs, v=v_bcs),
                            background_fields = (; b=B_field))

noise(x, z) = 1e-3 * randn() * exp(-(10z)^2/grid.Lz^2)
set!(model, u=noise, w=noise)

# The bottom-intensified noise above should accelerate the emergence of turbulence close to the
# wall.
#
# We are now ready to create the simulation. We begin by setting the initial time step
# conservatively, based on the smallest grid size of our domain and set-up a 

using Oceananigans.Units

simulation = Simulation(model, Δt = 0.5 * minimum_zspacing(grid) / V∞, stop_time = 12hours)

# We use `TimeStepWizard` to maximize Δt

wizard = TimeStepWizard(max_change=1.1, cfl=0.7)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))

# ## Model diagnostics
#
# We set-up a custom progress messenger using `Oceanostics.ProgressMessengers`, which allows
# us to combine different `ProgressMessenger`s into one:

using Oceanostics.ProgressMessengers

walltime_per_timestep = StepDuration() # This needs to instantiated here, and not in the function below
progress(simulation) = @info (PercentageProgress(with_prefix=false, with_units=false) + SimulationTime() + TimeStep() + MaxVelocities() + AdvectiveCFLNumber() + walltime_per_timestep)(simulation)

simulation.callbacks[:progress] = Callback(progress, IterationInterval(400))

# We now define some useful diagnostics for the flow. Namely, we define `RichardsonNumber`,
# `RossbyNumber` and `ErtelPotentialVorticity`:

using Oceanostics

b = model.tracers.b + model.background_fields.tracers.b
Ri = RichardsonNumber(model, model.velocities..., b)
Ro = RossbyNumber(model)
PV = ErtelPotentialVorticity(model, model.velocities..., b, model.coriolis)

# Note that the calculation of these quantities depends on the alignment with the true (geophysical)
# vertical and the rotation axis. Oceanostics already takes that into consideration by using
# `model.buoyancy` and `model.coriolis`, making their calculation much easier. Furthermore, passing
# the flag `add_background=true` automatically adds the `model`'s `BackgroundField`s to the resolved
# perturbations, which is important in our case for the correct calculation of ``\nabla b`` with the
# background stratification.
#
# Now we write these quantities to a NetCDF file:

output_fields = (; Ri, Ro, PV, b)

filename = "tilted_bottom_boundary_layer"
simulation.output_writers[:nc] = NetCDFWriter(model, output_fields,
                                                    filename = joinpath(@__DIR__, filename),
                                                    schedule = TimeInterval(20minutes),
                                                    overwrite_existing = true)

# ## Run the simulation and process results
#
# To run the simulation:

run!(simulation)

# Now we'll read the results and plot an animation

using Rasters

ds = RasterStack(simulation.output_writers[:nc].filepath)

# We now use Makie to create the figure and its axes

using CairoMakie

set_theme!(Theme(fontsize = 20))
fig = Figure()

kwargs = (xlabel="x [m]", ylabel="z [m]", height=150, width=250)
ax1 = Axis(fig[2, 1]; title = "Ri", kwargs...)
ax2 = Axis(fig[2, 2]; title = "Ro", kwargs...)
ax3 = Axis(fig[2, 3]; title = "PV", kwargs...);

# Next we an `Observable` to lift the values at each specific time and plot
# heatmaps, along with their colorbars, with buoyancy contours on top

n = Observable(1)

xC = Array(dims(ds, :xC))
xF = Array(dims(ds, :xF))
zC = Array(dims(ds, :zC))
zF = Array(dims(ds, :zF))

bₙ = @lift Array(ds.b[Ti=$n, yC=Near(0)])

Riₙ = @lift Array(ds.Ri[Ti=$n, yC=Near(0)])
hm1 = heatmap!(ax1, xC, zF, Riₙ; colormap = :coolwarm, colorrange = (-1, +1))
contour!(ax1, xC, zC, bₙ; levels=10, color=:white, linestyle=:dash, linewidth=0.5)
Colorbar(fig[3, 1], hm1, vertical=false, height=8, ticklabelsize=14)

Roₙ = @lift Array(ds.Ro[Ti=$n, yF=Near(0)])
hm2 = heatmap!(ax2, xF, zF, Roₙ; colormap = :balance, colorrange = (-10, +10))
contour!(ax2, xC, zC, bₙ; levels=10, color=:black, linestyle=:dash, linewidth=0.5)
Colorbar(fig[3, 2], hm2, vertical=false, height=8, ticklabelsize=14)

PVₙ = @lift Array(ds.PV[Ti=$n, yF=Near(0)])
hm3 = heatmap!(ax3, xF, zF, PVₙ; colormap = :coolwarm, colorrange = N²*f₀.*(-1.5, +1.5))
contour!(ax3, xC, zC, bₙ; levels=10, color=:white, linestyle=:dash, linewidth=0.5)
Colorbar(fig[3, 3], hm3, vertical=false, height=8, ticklabelsize=14);

# Now we mark the time by placing a vertical line in the bottom panel and adding a helpful title

times = dims(ds, :Ti)
title = @lift "Time = " * string(prettytime(times[$n]))
fig[1, 1:3] = Label(fig, title, fontsize=24, tellwidth=false);

# Finally, we adjust the figure dimensions to fit all the panels and record a movie

resize_to_layout!(fig)

@info "Animating..."
record(fig, filename * ".mp4", 1:length(times), framerate=10) do i
       n[] = i
end

