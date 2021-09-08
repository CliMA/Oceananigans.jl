# # Eady turbulence example
#
# In this example, we initialize a random velocity field and observe its viscous,
# turbulent decay in a two-dimensional domain. This example demonstrates:
#
#   * How to use a tuple of turbulence closures
#   * How to use hyperdiffusivity
#   * How to implement background velocity and tracer distributions
#   * How to use `ComputedField`s for output
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, JLD2, Plots"
# ```

# ## The Eady problem
#
# The "Eady problem" simulates the baroclinic instability problem proposed by Eric Eady in
# the classic paper
# ["Long waves and cyclone waves," Tellus (1949)](https://doi.org/10.3402/tellusa.v1i3.8507).
# The Eady problem is a simple, canonical model for the generation of mid-latitude
# atmospheric storms and the ocean eddies that permeate the world sea.
#
# In the Eady problem, baroclinic motion and turublence is generated by the interaction
# between turbulent motions and a stationary, geostrophically-balanced basic state that
# is unstable to baroclinic instability. In this example, the baroclinic generation of
# turbulence due to extraction of energy from the geostrophic basic state
# is balanced by a bottom boundary condition that extracts momentum from turbulent motions
# and serves as a crude model for the drag associated with an unresolved and small-scale
# turbulent bottom boundary layer.
#
# ### The geostrophic basic state
#
# The geostrophic basic state in the Eady problem is represented by the streamfunction,
#
# ```math
# ψ(y, z) = - α y (z + L_z) \, ,
# ```
#
# where ``α`` is the geostrophic shear and ``L_z`` is the depth of the domain.
# The background buoyancy includes both the geostrophic flow component,
# ``f ∂_z ψ``, where ``f`` is the Coriolis parameter, and a background stable stratification
# component, ``N^2 z``, where ``N`` is the buoyancy frequency:
#
# ```math
# B(y, z) = f ∂_z ψ + N^2 z = - α f y + N^2 z \, .
# ```
#
# The background velocity field is related to the geostrophic streamfunction via
# ``U = - ∂_y ψ`` such that
#
# ```math
# U(z) = α (z + L_z) \, .
# ```
#
# ### Boundary conditions
#
# All fields are periodic in the horizontal directions.
# We use "insulating", or zero-flux boundary conditions on the buoyancy perturbation
# at the top and bottom. We thus implicitly assume that the background vertical density
# gradient, ``N^2 z``, is maintained by a process external to our simulation.
# We use free-slip, or zero-flux boundary conditions on ``u`` and ``v`` at the surface
# where ``z=0``. At the bottom, we impose a momentum flux that extracts momentum and
# energy from the flow.
#
# #### Bottom boundary condition: quadratic bottom drag
#
# We model the effects of a turbulent bottom boundary layer on the eddy momentum budget
# with quadratic bottom drag. A quadratic cottom drag is introduced by imposing a vertical flux
# of horizontal momentum that removes momentum from the layer immediately above: in other words,
# the flux is negative (downwards) when the velocity at the bottom boundary is positive, and
# positive (upwards) with the velocity at the bottom boundary is negative.
# This drag term is "quadratic" because the rate at which momentum is removed is proportional
# to ``\boldsymbol{u} |\boldsymbol{u}|``, where ``\boldsymbol{u} = u \boldsymbol{\hat{x}} + 
# v \boldsymbol{\hat{y}}`` is the horizontal velocity.
#
# The ``x``-component of the quadratic bottom drag is thus
#
# ```math
# \tau_{xz}(z=L_z) = - c^D u \sqrt{u^2 + v^2} \, ,
# ```
#
# while the ``y``-component is
#
# ```math
# \tau_{yz}(z=L_z) = - c^D v \sqrt{u^2 + v^2} \, ,
# ```
#
# where ``c^D`` is a dimensionless drag coefficient and ``\tau_{xz}(z=L_z)`` and ``\tau_{yz}(z=L_z)``
# denote the flux of ``u`` and ``v`` momentum at ``z = L_z``, the bottom of the domain.
#
# ### Vertical and horizontal viscosity and diffusivity
#
# Vertical and horizontal viscosities and diffusivities are required
# to stabilize the Eady problem and can be idealized as modeling the effect of
# turbulent mixing below the grid scale. For both tracers and velocities we use
# a Laplacian vertical diffusivity ``κ_z ∂_z^2 c`` and a horizontal
# hyperdiffusivity ``ϰ_h (∂_x^4 + ∂_y^4) c``.
#
# ### Eady problem summary and parameters
#
# To summarize, the Eady problem parameters along with the values we use in this example are
#
# | Parameter name | Description | Value | Units |
# |:--------------:|:-----------:|:-----:|:-----:|
# | ``f``          | Coriolis parameter | ``10^{-4}`` | ``\mathrm{s^{-1}}`` |
# | ``N``          | Buoyancy frequency (square root of ``\partial_z B``) | ``10^{-3}`` | ``\mathrm{s^{-1}}`` |
# | ``\alpha``     | Background vertical shear ``\partial_z U`` | ``10^{-3}`` | ``\mathrm{s^{-1}}`` |
# | ``c^D``        | Bottom quadratic drag coefficient | ``10^{-4}`` | none |
# | ``κ_z``        | Laplacian vertical diffusivity | ``10^{-2}`` | ``\mathrm{m^2 s^{-1}}`` |
# | ``ϰ_h``        | Biharmonic horizontal diffusivity | ``10^{-2} \times \Delta x^4 / \mathrm{day}`` | ``\mathrm{m^4 s^{-1}}`` |
#
# We start off by importing `Oceananigans`, `Printf`, and some convenient constants
# for specifying dimensional units:

using Printf
using Oceananigans
using Oceananigans.Units: hours, day, days

# ## The grid
#
# We use a three-dimensional grid with a depth of 4000 m and a
# horizontal extent of 1000 km, appropriate for mesoscale ocean dynamics
# with characteristic scales of 50-200 km.

grid = RegularRectilinearGrid(size=(48, 48, 16), x=(0, 1e6), y=(0, 1e6), z=(-4e3, 0))

# ## Rotation
#
# The classical Eady problem is posed on an ``f``-plane. We use a Coriolis parameter
# typical to mid-latitudes on Earth,

coriolis = FPlane(f=1e-4) # [s⁻¹]

# ## The background flow
#
# We build a `NamedTuple` of parameters that describe the background flow,

basic_state_parameters = ( α = 10 * coriolis.f, # s⁻¹, geostrophic shear
                           f = coriolis.f,      # s⁻¹, Coriolis parameter
                           N = 1e-3,            # s⁻¹, buoyancy frequency
                          Lz = grid.Lz)         # m, ocean depth

# and then construct the background fields ``U`` and ``B``

## Background fields are defined via functions of x, y, z, t, and optional parameters
U(x, y, z, t, p) = + p.α * (z + p.Lz)
B(x, y, z, t, p) = - p.α * p.f * y + p.N^2 * z

U_field = BackgroundField(U, parameters=basic_state_parameters)
B_field = BackgroundField(B, parameters=basic_state_parameters)

# ## Boundary conditions
#
# The boundary conditions prescribe a quadratic drag at the bottom as a flux
# condition.

cᴰ = 1e-4 # quadratic drag coefficient

@inline drag_u(x, y, t, u, v, cᴰ) = - cᴰ * u * sqrt(u^2 + v^2)
@inline drag_v(x, y, t, u, v, cᴰ) = - cᴰ * v * sqrt(u^2 + v^2)

drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=cᴰ)
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=cᴰ)

u_bcs = FieldBoundaryConditions(bottom = drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom = drag_bc_v)

# ## Turbulence closures
#
# We use a horizontal hyperdiffusivity and a Laplacian vertical diffusivity
# to dissipate energy in the Eady problem.
# To use both of these closures at the same time, we set the keyword argument
# `closure` to a tuple of two closures.

κ₂z = 1e-2 # [m² s⁻¹] Laplacian vertical viscosity and diffusivity
κ₄h = 1e-1 / day * grid.Δx^4 # [m⁴ s⁻¹] horizontal hyperviscosity and hyperdiffusivity

Laplacian_vertical_diffusivity = AnisotropicDiffusivity(νh=0, κh=0, νz=κ₂z, κz=κ₂z)
biharmonic_horizontal_diffusivity = AnisotropicBiharmonicDiffusivity(νh=κ₄h, κh=κ₄h)

# ## Model instantiation
#
# We instantiate the model with the fifth-order WENO advection scheme, a 3rd order
# Runge-Kutta time-stepping scheme, and a `BuoyancyTracer`.

model = NonhydrostaticModel(
           architecture = CPU(),
                   grid = grid,
              advection = WENO5(),
            timestepper = :RungeKutta3,
               coriolis = coriolis,
                tracers = :b,
               buoyancy = BuoyancyTracer(),
      background_fields = (b=B_field, u=U_field),
                closure = (Laplacian_vertical_diffusivity, biharmonic_horizontal_diffusivity),
    boundary_conditions = (u=u_bcs, v=v_bcs)
)

# ## Initial conditions
#
# We seed our initial conditions with random noise stimulate the growth of
# baroclinic instability.

## A noise function, damped at the top and bottom
Ξ(z) = randn() * z/grid.Lz * (z/grid.Lz + 1)

## Scales for the initial velocity and buoyancy
Ũ = 1e-1 * basic_state_parameters.α * grid.Lz
B̃ = 1e-2 * basic_state_parameters.α * coriolis.f

uᵢ(x, y, z) = Ũ * Ξ(z)
vᵢ(x, y, z) = Ũ * Ξ(z)
bᵢ(x, y, z) = B̃ * Ξ(z)

set!(model, u=uᵢ, v=vᵢ, b=bᵢ)

# We subtract off any residual mean velocity to avoid exciting domain-scale
# inertial oscillations. We use a `sum` over the entire `parent` arrays or data
# to ensure this operation is efficient on the GPU (set `architecture = GPU()`
# in `NonhydrostaticModel` constructor to run this problem on the GPU if one
# is available).

ū = sum(model.velocities.u.data.parent) / (grid.Nx * grid.Ny * grid.Nz)
v̄ = sum(model.velocities.v.data.parent) / (grid.Nx * grid.Ny * grid.Nz)

model.velocities.u.data.parent .-= ū
model.velocities.v.data.parent .-= v̄
nothing # hide

# ## Simulation set-up
#
# We set up a simulation that runs for 10 days with a `JLD2OutputWriter` that saves the
# vertical vorticity and divergence every 2 hours. We limit the time-step to
# the maximum allowable due either to diffusion, internal waves, or advection by the background flow.

## Calculate absolute limit on time-step using diffusivities and
## background velocity.
Ū = basic_state_parameters.α * grid.Lz

max_Δt = min(grid.Δx / Ū, grid.Δx^4 / κ₄h, grid.Δz^2 / κ₂z, 1/basic_state_parameters.N)

simulation = Simulation(model, Δt = max_Δt, stop_time = 8days)

# ### The `TimeStepWizard`
#
# The TimeStepWizard manages the time-step adaptively, keeping the
# Courant-Freidrichs-Lewy (CFL) number close to `1.0` while ensuring
# the time-step does not increase beyond the maximum allowable value
# for numerical stability given the specified background flow, internal wave
# time scales, and diffusion time scales.

wizard = TimeStepWizard(cfl=0.85, Δt=max_Δt, max_change=1.1, max_Δt=max_Δt)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# ### A progress messenger
#
# We add a callback that prints out a helpful progress message while the simulation runs.

CFL = AdvectiveCFL(wizard)

start_time = time_ns()

progress(sim) = @printf("i: % 6d, sim time: % 10s, wall time: % 10s, Δt: % 10s, CFL: %.2e\n",
                        sim.model.clock.iteration,
                        prettytime(sim.model.clock.time),
                        prettytime(1e-9 * (time_ns() - start_time)),
                        prettytime(sim.Δt),
                        CFL(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

# ### Output
#
# To visualize the baroclinic turbulence ensuing in the Eady problem,
# we use `ComputedField`s to diagnose and output vertical vorticity and divergence.
# Note that `ComputedField`s take "AbstractOperations" on `Field`s as input:

u, v, w = model.velocities # unpack velocity `Field`s

## Vertical vorticity [s⁻¹]
ζ = ComputedField(∂x(v) - ∂y(u))

## Horizontal divergence, or ∂x(u) + ∂y(v) [s⁻¹]
δ = ComputedField(-∂z(w))

# With the vertical vorticity, `ζ`, and the horizontal divergence, `δ` in hand,
# we create a `JLD2OutputWriter` that saves `ζ` and `δ` and add them to
# `simulation`.

simulation.output_writers[:fields] = JLD2OutputWriter(model, (ζ=ζ, δ=δ),
                                                      schedule = TimeInterval(4hours),
                                                        prefix = "eady_turbulence",
                                                         force = true)
nothing # hide

# All that's left is to press the big red button:

run!(simulation)

# ## Visualizing Eady turbulence
#
# We animate the results by opening the JLD2 file, extracting data for
# the iterations we ended up saving at, and ploting slices of the saved
# fields. We prepare for animating the flow by creating coordinate arrays,
# opening the file, building a vector of the iterations that we saved
# data at, and defining a function for computing colorbar limits:

using JLD2, Plots

## Coordinate arrays
xζ, yζ, zζ = nodes(ζ)
xδ, yδ, zδ = nodes(δ)

## Open the file with our data
file = jldopen(simulation.output_writers[:fields].filepath)

## Extract a vector of iterations
iterations = parse.(Int, keys(file["timeseries/t"]))

# This utility is handy for calculating nice contour intervals:

function nice_divergent_levels(c, clim, nlevels=31)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
    return levels
end
nothing # hide

# Now we're ready to animate.

@info "Making an animation from saved data..."

anim = @animate for (i, iter) in enumerate(iterations)

    ## Load 3D fields from file
    t = file["timeseries/t/$iter"]
    R_snapshot = file["timeseries/ζ/$iter"] ./ coriolis.f
    δ_snapshot = file["timeseries/δ/$iter"]

    surface_R = R_snapshot[:, :, grid.Nz]
    surface_δ = δ_snapshot[:, :, grid.Nz]

    slice_R = R_snapshot[:, 1, :]
    slice_δ = δ_snapshot[:, 1, :]

    Rlim = 0.5 * maximum(abs, R_snapshot) + 1e-9
    δlim = 0.5 * maximum(abs, δ_snapshot) + 1e-9

    Rlevels = nice_divergent_levels(R_snapshot, Rlim)
    δlevels = nice_divergent_levels(δ_snapshot, δlim)

    @info @sprintf("Drawing frame %d from iteration %d: max(ζ̃ / f) = %.3f \n",
                   i, iter, maximum(abs, surface_R))

    xy_kwargs = (xlims = (0, grid.Lx), ylims = (0, grid.Lx),
                 xlabel = "x (m)", ylabel = "y (m)",
                 aspectratio = 1,
                 linewidth = 0, color = :balance, legend = false)

    xz_kwargs = (xlims = (0, grid.Lx), ylims = (-grid.Lz, 0),
                 xlabel = "x (m)", ylabel = "z (m)",
                 aspectratio = grid.Lx / grid.Lz * 0.5,
                 linewidth = 0, color = :balance, legend = false)

    R_xy = contourf(xζ, yζ, surface_R'; clims=(-Rlim, Rlim), levels=Rlevels, xy_kwargs...)
    δ_xy = contourf(xδ, yδ, surface_δ'; clims=(-δlim, δlim), levels=δlevels, xy_kwargs...)
    R_xz = contourf(xζ, zζ, slice_R'; clims=(-Rlim, Rlim), levels=Rlevels, xz_kwargs...)
    δ_xz = contourf(xδ, zδ, slice_δ'; clims=(-δlim, δlim), levels=δlevels, xz_kwargs...)

    plot(R_xy, δ_xy, R_xz, δ_xz,
           size = (1000, 800),
           link = :x,
         layout = Plots.grid(2, 2, heights=[0.5, 0.5, 0.2, 0.2]),
          title = [@sprintf("ζ(t=%s) / f", prettytime(t)) @sprintf("δ(t=%s) (s⁻¹)", prettytime(t)) "" ""])

    iter == iterations[end] && close(file)
end

mp4(anim, "eady_turbulence.mp4", fps = 8) # hide
