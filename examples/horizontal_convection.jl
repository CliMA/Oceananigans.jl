# # Horizontal convection example
#
# In this example, we initialize flow at rest under the effect of gravity and in which a 
# non-uniform buoyancy along the top surface is imposed. This problems setup is referred to 
# as "horizontal convection".
#
# This example demonstrates:
#
#   * How to use `ComputedField`s for output
#   * How to post-process saved output
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, JLD2, Plots"
# ```

# ## Horizontal convection
#
# We'll consider here the two-dimensional verion of horizontal convection in which a flow
# ``\boldsymbol{u} = (u, w)`` on the ``(x, z)``-plane is evolved under the effect of gravity
# and with a prescribed, non-uniform buoyancy at the top-surface of the domain.
#
# ```math
# \begin{gather}
# \partial_t \boldsymbol{u} + \boldsymbol{u \cdot \nabla} \boldsymbol{u} + \boldsymbol{\nabla}p = b \hat{\boldsymbol{z}} + \nu \nabla^2 \boldsymbol{u} \, , \\
# \partial_t b + \boldsymbol{u \cdot \nabla} b = \kappa \nabla^2 b \, , \\
# \boldsymbol{\nabla \cdot u} = 0 \, .
# \end{gather}
# ```
# 
# The domain here is ``-L_x/2 \le x \le L_x/2`` and ``-H \le z \le 0``.
#
# ### Boundary conditions
#
# At the surface (``z = 0``), the imposed buoyancy is ``b_s(x, z = 0, t) = - b_* \cos(2 \pi x / L_x)``
# while buoyancy satisfies zero-flux boundary conditions on all other boundaries. We use 
# free-slip boundary conditions on ``u`` and ``w`` at all boundaries.
#
# ### Non-dimensional control parameters
#
# The problem is characterized by three non-dimensional parameters. The first is the domain's
# aspect ratio, ``A = \frac{L_x}{H}`` and the rest two are the Rayleigh and Prandtl numbers:
#
# ```math
# Ra = \frac{L_x^3 b_*}{\nu \kappa} \, , \quad \text{and}\, \quad Pr = \frac{\nu}{\kappa} \, .
# ```
#
# For a domain with a given extent, the nondimensional values of ``Ra`` and ``Pr`` uniquely
# prescribe the viscosity and diffusivity,
# 
# ```math
# \nu = \sqrt{\frac{Pr b_* L_x^3}{Ra}} \quad \text{and} \quad \kappa = \sqrt{\frac{b_* L_x^3}{Pr Ra}} \, .
# ```
#
# We start off by importing `Oceananigans` and `Printf`.

using Printf
using Oceananigans

# ## The grid
#
# We use a two-dimensional grid with an aspect ratio ``A = L_x / H = 2``.

H = 1.0         # vertical domain extent
Lx = 2H         # horizontal domain extent
Nx, Nz = 128, 64 # horizontal, vertical resolution

grid = RegularRectilinearGrid(size = (Nx, Nz),
                                 x = (-Lx/2, Lx/2),
                                 z = (-H, 0),
                              halo = (3, 3),
                          topology = (Bounded, Flat, Bounded))

# Any attempts for `VerticallyStretchedRectilinearGrid` failed...
# 
# ```
# σ = 0.2  # stretching factor
# 
# hyperbolically_spaced_faces(k) = H - H * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ))
# 
# grid = VerticallyStretchedRectilinearGrid(size = (Nx, Nz),
#                                           topology = (Bounded, Flat, Bounded),
#                                           x=(-Lx/2, Lx/2),
#                                           halo = (3, 3),
#                                           z_faces = hyperbolically_spaced_faces)
# 
# # We plot vertical spacing versus depth to inspect the prescribed grid stretching:
# using Plots
# 
# plot(grid.Δzᵃᵃᶜ[1:Nz], grid.zᵃᵃᶜ[1:Nz],
#      marker = :circle,
#      ylabel = "Depth",
#      xlabel = "Vertical spacing",
#      legend = nothing)
# ```

# ## Boundary conditions
#
# We impose a buoyancy boundary condition
#
# ```math
# b(x, z=0, t) = - b_* \cos (2π x / L_x) \, .
# ```

b★ = 1.0

@inline bₛ(x, y, t) = - cos(π * x)

b_bcs = TracerBoundaryConditions(grid,
                                 top = ValueBoundaryCondition(bₛ))

# ## Turbulence closures
#
# We use isotropic viscosity and diffusivities, `ν` and `κ` whose values are obtain from the
# prescribed ``Ra`` and ``Pr`` numbers. Here, we use ``Pr = 1`` and ``Ra = 10^8``:

Pr = 1.0   # The Prandtl number
Ra = 1e8   # The Rayleigh number

ν = sqrt(Pr * b★ * Lx^3 / Ra)  # Laplacian viscosity
κ = ν * Pr                     # Laplacian diffusivity
nothing # hide

# ## Model instantiation
#
# We instantiate the model with the fifth-order WENO advection scheme, a 3rd order
# Runge-Kutta time-stepping scheme, and a `BuoyancyTracer`.

model = IncompressibleModel(
           architecture = CPU(),
                   grid = grid,
              advection = WENO5(),
            timestepper = :RungeKutta3,
                tracers = :b,
               buoyancy = BuoyancyTracer(),
                closure = IsotropicDiffusivity(ν=ν, κ=κ),
    boundary_conditions = (b=b_bcs,)
    )

# ## Simulation set-up
#
# We set up a simulation that runs up to ``t = 40`` with a `JLD2OutputWriter` that saves the
# flowspeed, ``\sqrt{u^2 + w^2}``, the vorticity, ``\partial_z u - \partial_x w``, and the 
# buoyancy dissipation, ``\kappa |\boldsymbol{\nabla} b|^2``.
#
# ### The `TimeStepWizard`
#
# The TimeStepWizard manages the time-step adaptively, keeping the Courant-Freidrichs-Lewy 
# (CFL) number close to `0.75` while ensuring the time-step does not increase beyond the 
# maximum allowable value for numerical stability.

max_Δt = 4e-2
wizard = TimeStepWizard(cfl=0.75, Δt=5e-3, max_change=1.2, max_Δt=max_Δt)

# ### A progress messenger
#
# We write a function that prints out a helpful progress message while the simulation runs.

CFL = AdvectiveCFL(wizard)

start_time = time_ns()

progress(sim) = @printf("i: % 6d, sim time: % 10s, wall time: % 10s, Δt: % 10s, CFL: %.2e\n",
                        sim.model.clock.iteration,
                        prettytime(sim.model.clock.time),
                        prettytime(1e-9 * (time_ns() - start_time)),
                        prettytime(sim.Δt.Δt),
                        CFL(sim.model))
nothing # hide

# ### Build the simulation
#
# We're ready to build and run the simulation. We ask for a progress message and time-step update
# every 100 iterations,

simulation = Simulation(model, Δt = wizard, iteration_interval = 50,
                                                     stop_time = 40.05,
                                                      progress = progress)

# ### Output
#
# We use `ComputedField`s to diagnose and output the total flow speed, the vorticity, ``\zeta``,
# and the buoyancy dissipation, ``\chi = \kappa |\boldsymbol{\nabla}b|^2``. Note that 
# `ComputedField`s take "AbstractOperations"on `Field`s as input:

u, v, w = model.velocities # unpack velocity `Field`s
b = model.tracers.b        # unpack buoyancy `Field`

## total flow speed
speed = ComputedField(sqrt(u^2 + w^2))

## y-component of vorticity
ζ = ComputedField(∂z(u) - ∂x(w))

## buoyancy dissipation
χ_op = @at (Center, Center, Center) κ * (∂x(b)^2 + ∂z(b)^2)
χ = ComputedField(χ_op)

outputs = (s = speed, b = b, ζ = ζ, χ = χ)
nothing # hide

# We create a `JLD2OutputWriter` that saves the speed, the vorticity, and the buoyancy dissipation.
# We then add the `JLD2OutputWriter` to the `simulation`.

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(0.1),
                                                        prefix = "horizontal_convection",
                                                         force = true)
nothing # hide

# Ready to press the big red button:

run!(simulation)

# ## Visualizing horizontal convection
#
# We animate the results by opening the JLD2 file, extracting data for the iterations we ended
# up saving at, and ploting the saved fields. We prepare for animating the flow by creating 
# coordinate arrays, opening the file, building a vector of the iterations that we saved data at,
# and defining a function for computing colorbar limits:

using JLD2, Plots

## Coordinate arrays
xs, ys, zs = nodes(speed)
xb, yb, zb = nodes(b)
xζ, yζ, zζ = nodes(ζ)
xχ, yχ, zχ = nodes(χ)

## Open the file with our data
file = jldopen(simulation.output_writers[:fields].filepath)

## Extract a vector of iterations
iterations = parse.(Int, keys(file["timeseries/t"]))

# The utilities below come handy for calculating nice contour intervals:

function nice_levels(c, clim, nlevels=41)
    levels = range(0, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    clim < cmax && (levels = vcat(levels, [cmax]))
    return levels
end

function nice_divergent_levels(c, clim, nlevels=41)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
    return levels
end

nothing # hide

# Now we're ready to animate.

@info "Making an animation from saved data..."

anim = @animate for (i, iter) in enumerate(iterations)

    ## Load fields from file
    t = file["timeseries/t/$iter"]
    s_snapshot = file["timeseries/s/$iter"][:, 1, :]
    b_snapshot = file["timeseries/b/$iter"][:, 1, :]
    ζ_snapshot = file["timeseries/ζ/$iter"][:, 1, :]
    χ_snapshot = file["timeseries/χ/$iter"][:, 1, :]
    
    ## determine colorbar limits and contour levels
    slim = 0.6
    slevels = vcat(range(0, stop=slim, length=31), [0.6])

    blim = 0.6
    blevels = vcat([-1], range(-blim, stop=blim, length=51), [1])
    
    ζlim = 9
    ζlevels = nice_divergent_levels(ζ_snapshot, ζlim)

    χlim = 0.025
    χlevels = nice_levels(χ_snapshot, χlim)

    @info @sprintf("Drawing frame %d from iteration %d:", i, iter)

    kwargs = (      xlims = (-Lx/2, Lx/2),
                    ylims = (-H, 0),
                   xlabel = "x / H",
                   ylabel = "z / H",
              aspectratio = 1,
                linewidth = 0)

    s_plot = contourf(xs, zs, s_snapshot';
                      clims = (0, slim), levels = slevels,
                      color = :speed, kwargs...)
    s_title = @sprintf("speed √[(u²+w²)/(b⋆H)] @ t=%1.2f", t)
    
    b_plot = contourf(xb, zb, b_snapshot';
                      clims = (-blim, blim), levels = blevels,
                      color = :thermal, kwargs...)
    b_title = @sprintf("buoyancy, b/b⋆ @ t=%1.2f", t)
    
    ζ_plot = contourf(xζ, zζ, ζ_snapshot';
                      clims=(-ζlim, ζlim), levels = ζlevels,
                      color = :balance, kwargs...)
    ζ_title = @sprintf("vorticity, (∂u/∂z - ∂w/∂x) √(H/b⋆) @ t=%1.2f", t)
    
    χ_plot = contourf(xχ, zχ, χ_snapshot';
                      clims = (0, χlim), levels = χlevels,
                      color = :dense, kwargs...)
    χ_title = @sprintf("buoyancy dissipation, κ|∇b|² √(H/b⋆⁵) @ t=%1.2f", t)
    
    plot(s_plot, b_plot, ζ_plot, χ_plot,
            dpi = 120,
           size = (700.25, 1200.25),
           link = :x,
         layout = Plots.grid(4, 1),
          title = [s_title, b_title, ζ_title, χ_title])

    iter == iterations[end] && close(file)
end

mp4(anim, "horizontal_convection.mp4", fps = 16) # hide
