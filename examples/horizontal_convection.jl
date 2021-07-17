# # Horizontal convection example
#
# In "horizontal convection", a non-uniform buoyancy is imposed on top of an initially resting fluid.
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
# We consider here two-dimensional horizontal convection of an incompressible flow ```\boldsymbol{u} = (u, w)``
# on the ``(x, z)``-plane (``-L_x/2 \le x \le L_x/2`` and ``-H \le z \le 0``). The flow evolves
# under the effect of gravity. The only forcing on the fluid comes from a prescribed, non-uniform
# buoyancy at the top-surface of the domain.
#
# We start by importing `Oceananigans` and `Printf`.

using Oceananigans
using Printf

# ### The grid

H = 1.0          # vertical domain extent
Lx = 2H          # horizontal domain extent
Nx, Nz = 128, 64 # horizontal, vertical resolution

grid = RegularRectilinearGrid(size = (Nx, Nz),
                                 x = (-Lx/2, Lx/2),
                                 z = (-H, 0),
                              halo = (3, 3),
                          topology = (Bounded, Flat, Bounded))

# ### Boundary conditions
#
# At the surface, the imposed buoyancy is
# ```math
# b(x, z = 0, t) = - b_* \cos (2 \pi x / L_x) \, ,
# ```
# while zero-flux boundary conditions are imposed on all other boundaries. We use free-slip 
# boundary conditions on ``u`` and ``w`` everywhere.

b★ = 1.0q
k = 2π / Lx

@inline bˢ(x, y, t, p) = - p.b★ * cos(p.k * x)

b_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(bˢ, parameters=(; b★, k)))

# ### Non-dimensional control parameters and Turbulence closure
#
# The problem is characterized by three non-dimensional parameters. The first is the domain's
# aspect ratio, ``A = L_x / H`` and the other two are the Rayleigh (``Ra``) and Prandtl (``Pr``)
# numbers:
#
# ```math
# Ra = \frac{L_x^3 b_*}{\nu \kappa} \, , \quad \text{and}\, \quad Pr = \frac{\nu}{\kappa} \, .
# ```
#
# The Prandtl number expresses the ratio of momentum over heat diffusion; the Rayleigh number
# is a measure of the relative importance of gravity over viscosity in the momentum equation.
#
# For a domain with a given extent, the nondimensional values of ``Ra`` and ``Pr`` uniquely
# determine the viscosity and diffusivity, i.e.,
# 
# ```math
# \nu = \sqrt{\frac{Pr b_* L_x^3}{Ra}} \quad \text{and} \quad \kappa = \sqrt{\frac{b_* L_x^3}{Pr Ra}} \, .
# ```
#
# We use isotropic viscosity and diffusivities, `ν` and `κ` whose values are obtain from the
# prescribed ``Ra`` and ``Pr`` numbers. Here, we use ``Pr = 1`` and ``Ra = 10^8``:

Pr = 1.0    # Prandtl number
Ra = 1e8    # Rayleigh number

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
# We set up a simulation that runs up to ``t = 40`` with a `JLD2OutputWriter` that saves the flow
# speed, ``\sqrt{u^2 + w^2}``, the buoyancy, ``b``, andthe vorticity, ``\partial_z u - \partial_x w``.
#
# ### The `TimeStepWizard`
#
# The TimeStepWizard manages the time-step adaptively, keeping the Courant-Freidrichs-Lewy 
# (CFL) number close to `0.75` while ensuring the time-step does not increase beyond the 
# maximum allowable value for numerical stability.

max_Δt = 1e-1
wizard = TimeStepWizard(cfl=0.75, Δt=1e-2, max_change=1.2, max_Δt=max_Δt)

# ### A progress messenger
#
# We write a function that prints out a helpful progress message while the simulation runs.

CFL = AdvectiveCFL(wizard)

start_time = time_ns()

progress(sim) = @printf("i: % 6d, sim time: % 1.3f, wall time: % 10s, Δt: % 1.4f, CFL: %.2e\n",
                        sim.model.clock.iteration,
                        sim.model.clock.time,
                        prettytime(1e-9 * (time_ns() - start_time)),
                        sim.Δt.Δt,
                        CFL(sim.model))
nothing # hide

# ### Build the simulation
#
# We're ready to build and run the simulation. We ask for a progress message and time-step update
# every 50 iterations,

simulation = Simulation(model, Δt = wizard, iteration_interval = 50,
                                                     stop_time = 40.0,
                                                      progress = progress)

# ### Output
#
# We use `ComputedField`s to diagnose and output the total flow speed, the vorticity, ``ζ``,
# and the buoyancy dissipation, ``χ = κ |∇b|²``. Note that 
# `ComputedField`s take "AbstractOperations" on `Field`s as input:

u, v, w = model.velocities # unpack velocity `Field`s
b = model.tracers.b        # unpack buoyancy `Field`

## total flow speed
s = ComputedField(sqrt(u^2 + w^2))

## y-component of vorticity
ζ = ComputedField(∂z(u) - ∂x(w))

outputs = (s = s, b = b, ζ = ζ)
nothing # hide

# We create a `JLD2OutputWriter` that saves the speed, and the vorticity.
# We then add the `JLD2OutputWriter` to the `simulation`.

saved_output_prefix = "horizontal_convection"
saved_output_filename = saved_output_prefix * ".jld2"

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                  field_slicer = nothing,
                                                      schedule = TimeInterval(0.5),
                                                        prefix = saved_output_prefix,
                                                         force = true)
nothing # hide

# Ready to press the big red button:

run!(simulation)

# ## Load saved output, process, visualize
#
# We animate the results by opening the JLD2 file, extracting data for the iterations we ended
# up saving at, and ploting the saved fields. We prepare for animating the flow by creating 
# coordinate arrays, opening the file, building a vector of the iterations that we saved data at.

using JLD2, Plots
using Oceananigans
using Oceananigans.Fields
using Oceananigans.AbstractOperations: volume

saved_output_prefix = "horizontal_convection"
saved_output_filename = saved_output_prefix * ".jld2"

## Open the file with our data
file = jldopen(saved_output_filename)

κ = file["closure/κ/b"]

s_timeseries = FieldTimeSeries(saved_output_filename, "s")
b_timeseries = FieldTimeSeries(saved_output_filename, "b")
ζ_timeseries = FieldTimeSeries(saved_output_filename, "ζ")

times = b_timeseries.times

## Coordinate arrays
xs, ys, zs = nodes(s_timeseries[1])
xb, yb, zb = nodes(b_timeseries[1])
xζ, yζ, zζ = nodes(ζ_timeseries[1])
nothing # hide

# Now we're ready to animate.

@info "Making an animation from saved data..."

anim = @animate for i in 1:length(times)

    ## Load fields from `FieldTimeSeries` and compute χ
    t = times[i]
    
    s_snapshot = interior(s_timeseries[i])[:, 1, :]
    ζ_snapshot = interior(ζ_timeseries[i])[:, 1, :]
    
    b = b_timeseries[i]
    χ = ComputedField(κ * (∂x(b)^2 + ∂z(b)^2))
    compute!(χ)
    
    b_snapshot = interior(b)[:, 1, :]
    χ_snapshot = interior(χ)[:, 1, :]
    
    ## determine colorbar limits and contour levels
    slim = 0.6
    slevels = vcat(range(0, stop=slim, length=31), [slim])

    blim = 0.6
    blevels = vcat([-1], range(-blim, stop=blim, length=51), [1])
    
    ζlim = 9
    ζlevels = range(-ζlim, stop=ζlim, length=41)

    χlim = 0.025
    χlevels = range(0, stop=χlim, length=41)

    @info @sprintf("Drawing frame %d:", i)

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
    
    ζ_plot = contourf(xζ, zζ, clamp.(ζ_snapshot', -ζlim, ζlim);
                      clims=(-ζlim, ζlim), levels = ζlevels,
                      color = :balance, kwargs...)
    ζ_title = @sprintf("vorticity, (∂u/∂z - ∂w/∂x) √(H/b⋆) @ t=%1.2f", t)
    
    χ_plot = contourf(xs, zs, χ_snapshot';
                      clims = (0, χlim), levels = χlevels,
                      color = :dense, kwargs...)
    χ_title = @sprintf("buoyancy dissipation, κ|∇b|² √(H/b⋆⁵) @ t=%1.2f", t)
    
    plot(s_plot, b_plot, ζ_plot, χ_plot,
           size = (700, 1200),
           link = :x,
         layout = Plots.grid(4, 1),
          title = [s_title b_title ζ_title χ_title])
end

mp4(anim, "horizontal_convection.mp4", fps = 16) # hide

# At higher Rayleigh numbers the flow becomes much more vigorous. See, for example, an animation
# of the voricity of the fluid at ``Ra = 10^{12}`` on [vimeo](https://vimeo.com/573730711). 

# ### The Nusselt number
#
# Often we are interested on how much the flow enhances mixing. This is quantified by the
# Nusselt number, which measures how much the flow enhances mixing compared if only diffusion
# was in operation. The Nusselt number is given by
#
# ```math
# Nu = \frac{\langle \chi \rangle}{\langle \chi_{\rm diff} \rangle} \, ,
# ```
#
# where angle brackets above denote both a volume and time average and  ``\chi_{\rm diff}`` is
# the buoyancy dissipation that we get without any flow, i.e., the buoyancy dissipation associated
# with the buoyancy distribution that satisfies
#
# ```math
# \kappa \nabla^2 b_{\rm diff} = 0 \, ,
# ```
#
# with the same boundary conditions same as our setup. In this case we can readily find that
#
# ```math
# b_{\rm diff}(x, z) = b_s(x) \frac{\cosh \left [2 \pi (H + z) / L_x \right ]}{\cosh(2 \pi H / L_x)} \, ,
# ```
# which implies ``\langle \chi_{\rm diff} \rangle = \kappa b_*^2 \pi \tanh(2 \pi Η /Lx)``.
#
# We use the loaded `FieldTimeSeries` to compute the Nusselt number from buoyancy and the volume
# average kinetic energy of the fluid.
#
# First we compute the diffusive buoyancy dissipation, ``\chi_{\rm diff}`` (which is just a
# scalar):

H  = file["grid/Lz"]
Lx = file["grid/Lx"]
κ  = file["closure/κ/b"]

χ_diff = κ * b★^2 * π * tanh(2π * H / Lx)
nothing # hide

# We then create two `ReducedField`s to store the volume integrals of the kinetic energy density
# and the buoyancy dissipation. We need the `grid` to do so; the `grid` can be recoverd from
# any `FieldTimeSeries`, e.g.,

grid = b_timeseries.grid

∫ⱽ_s² = ReducedField(Nothing, Nothing, Nothing, CPU(), grid, dims=(1, 2, 3))
∫ⱽ_mod²_∇b = ReducedField(Nothing, Nothing, Nothing, CPU(), grid, dims=(1, 2, 3))

# We recover the time from the saved `FieldTimeSeries` and construct two empty arrays to store
# the volume-averaged kinetic energy and the instantaneous Nusselt number,

t = b_timeseries.times

kinetic_energy, Nu = zeros(length(t)), zeros(length(t))
nothing # hide

# Now we can loop over the fields in the `FieldTimeSeries`, compute `KineticEnergy` and `Nu`,
# and plot.

for i = 1:length(t)
    s = s_timeseries[i]
    sum!(∫ⱽ_s², s^2 * volume)
    kinetic_energy[i] = 0.5 * ∫ⱽ_s²[1, 1, 1]  / (Lx * H)
    
    b = b_timeseries[i]
    sum!(∫ⱽ_mod²_∇b, (∂x(b)^2 + ∂z(b)^2) * volume)
    Nu[i] = (κ *  ∫ⱽ_mod²_∇b[1, 1, 1]) / χ_diff
end

p1 = plot(t, kinetic_energy,
          xlabel = "time",
          ylabel = "KE / (b⋆H)",
       linewidth = 3,
          legend = :none)

p2 = plot(t, Nu,
          xlabel = "time",
          ylabel = "Nu",
       linewidth = 3,
          legend = :none)

plot(p1, p2, layout = Plots.grid(2, 1))
