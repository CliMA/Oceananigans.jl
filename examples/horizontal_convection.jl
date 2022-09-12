# # Horizontal convection example
#
# In "horizontal convection", a non-uniform buoyancy is imposed on top of an initially resting fluid.
#
# This example demonstrates:
#
#   * How to use computed `Field`s for output.
#   * How to post-process saved output using `FieldTimeSeries`.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, JLD2, CairoMakie"
# ```

# ## Horizontal convection
#
# We consider here two-dimensional horizontal convection of an incompressible flow ``\boldsymbol{u} = (u, w)``
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

grid = RectilinearGrid(size = (Nx, Nz),
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

b★ = 1.0

@inline bˢ(x, y, t, p) = - p.b★ * cos(2π * x / p.Lx)

b_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(bˢ, parameters=(; b★, Lx)))

# ### Non-dimensional control parameters and Turbulence closure
#
# The problem is characterized by three non-dimensional parameters. The first is the domain's
# aspect ratio, ``L_x / H`` and the other two are the Rayleigh (``Ra``) and Prandtl (``Pr``)
# numbers:
#
# ```math
# Ra = \frac{b_* L_x^3}{\nu \kappa} \, , \quad \text{and}\, \quad Pr = \frac{\nu}{\kappa} \, .
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

model = NonhydrostaticModel(; grid,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = ScalarDiffusivity(; ν, κ),
                            boundary_conditions = (; b=b_bcs))

# ## Simulation set-up
#
# We set up a simulation that runs up to ``t = 40`` with a `JLD2OutputWriter` that saves the flow
# speed, ``\sqrt{u^2 + w^2}``, the buoyancy, ``b``, andthe vorticity, ``\partial_z u - \partial_x w``.

simulation = Simulation(model, Δt=1e-2, stop_time=40.0)

# ### The `TimeStepWizard`
#
# The TimeStepWizard manages the time-step adaptively, keeping the Courant-Freidrichs-Lewy 
# (CFL) number close to `0.75` while ensuring the time-step does not increase beyond the 
# maximum allowable value for numerical stability.

wizard = TimeStepWizard(cfl=0.75, max_change=1.2, max_Δt=1e-1)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(50))

# ### A progress messenger
#
# We write a function that prints out a helpful progress message while the simulation runs.

progress(sim) = @printf("i: % 6d, sim time: % 1.3f, wall time: % 10s, Δt: % 1.4f, CFL: %.2e\n",
                        iteration(sim), time(sim), prettytime(sim.run_wall_time),
                        sim.Δt, AdvectiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(50))

# ### Output
#
# We use computed `Field`s to diagnose and output the total flow speed, the vorticity, ``\zeta``,
# and the buoyancy, ``b``. Note that computed `Field`s take "AbstractOperations" on `Field`s as
# input:

u, v, w = model.velocities # unpack velocity `Field`s
b = model.tracers.b        # unpack buoyancy `Field`

## total flow speed
s = sqrt(u^2 + w^2)

## y-component of vorticity
ζ = ∂z(u) - ∂x(w)
nothing # hide

# We create a `JLD2OutputWriter` that saves the speed, and the vorticity.
# We then add the `JLD2OutputWriter` to the `simulation`.

saved_output_filename = "horizontal_convection.jld2"

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; s, b, ζ),
                                                      schedule = TimeInterval(0.5),
                                                      filename = saved_output_filename,
                                                      overwrite_existing = true)
nothing # hide

# Ready to press the big red button:

run!(simulation)

# ## Load saved output, process, visualize
#
# We animate the results by opening the JLD2 file, extracting data for the iterations we ended
# up saving at, and ploting the saved fields. From the saved buoyancy field we compute the 
# buoyancy dissipation, ``\chi = \kappa |\boldsymbol{\nabla} b|^2``, and plot that also.
#
# To start we load the saved fields are `FieldTimeSeries` and prepare for animating the flow by
# creating coordinate arrays that each field lives on.

using CairoMakie
using Oceananigans
using Oceananigans.Fields
using Oceananigans.AbstractOperations: volume

saved_output_filename = "horizontal_convection.jld2"

## Open the file with our data
s_timeseries = FieldTimeSeries(saved_output_filename, "s")
b_timeseries = FieldTimeSeries(saved_output_filename, "b")
ζ_timeseries = FieldTimeSeries(saved_output_filename, "ζ")

times = b_timeseries.times

## Coordinate arrays
xs, ys, zs = nodes(s_timeseries[1])
xb, yb, zb = nodes(b_timeseries[1])
xζ, yζ, zζ = nodes(ζ_timeseries[1])
nothing # hide

χ_timeseries = deepcopy(b_timeseries)

for i in 1:length(times)
  bᵢ = b_timeseries[i]
  χ_timeseries[i] .= κ * (∂x(bᵢ)^2 + ∂z(bᵢ)^2)
end


# Now we're ready to animate using Makie.

@info "Making an animation from saved data..."

n = Observable(1)

title = @lift @sprintf("t=%1.2f", times[$n])

sₙ = @lift interior(s_timeseries[$n], :, 1, :)
ζₙ = @lift interior(ζ_timeseries[$n], :, 1, :)
bₙ = @lift interior(b_timeseries[$n], :, 1, :)
χₙ = @lift interior(χ_timeseries[$n], :, 1, :)

slim = 0.6
blim = 0.6
ζlim = 9
χlim = 0.025

axis_kwargs = (xlabel = L"x / H",
               ylabel = L"z / H",
               limits = ((-Lx/2, Lx/2), (-H, 0)),
               aspect = Lx / H,
               titlesize = 20)

fig = Figure(resolution = (600, 1100))

ax_s = Axis(fig[2, 1];
            title = L"speed, $(u^2+w^2)^{1/2} / (b_* H)^{1/2}", axis_kwargs...)

ax_b = Axis(fig[3, 1];
            title = L"buoyancy, $b / b_*$", axis_kwargs...)

ax_ζ = Axis(fig[4, 1];
            title = L"vorticity, $(∂u/∂z - ∂w/∂x) (H/b_*)^{1/2}$", axis_kwargs...)

ax_χ = Axis(fig[5, 1];
            title = L"buoyancy dissipation, $κ |\mathbf{\nabla}b|^2 (H/b_*^5)^{1/2}$", axis_kwargs...)

fig[1, :] = Label(fig, title, textsize=24, tellwidth=false)

hm_s = heatmap!(ax_s, xs, zs, sₙ;
                colorrange = (0, slim),
                colormap = :speed)
Colorbar(fig[2, 2], hm_s)

hm_b = heatmap!(ax_b, xb, zb, bₙ;
                colorrange = (-blim, blim),
                colormap = :thermal)
Colorbar(fig[3, 2], hm_b)

hm_ζ = heatmap!(ax_ζ, xζ, zζ, ζₙ;
                colorrange = (-ζlim, ζlim),
                colormap = :balance)
Colorbar(fig[4, 2], hm_ζ)

hm_χ = heatmap!(ax_χ, xs, zs, χₙ;
                colorrange = (0, χlim),
                colormap = :dense)
Colorbar(fig[5, 2], hm_χ)

# And, finally, we record a movie.

frames = 1:length(times)

record(fig, "horizontal_convection.mp4", frames, framerate=8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end
nothing #hide

# ![](horizontal_convection.mp4)


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
# where angle brackets above denote both a volume and time average and ``\chi_{\rm diff}`` is
# the buoyancy dissipation that we get without any flow, i.e., the buoyancy dissipation associated
# with the buoyancy distribution that satisfies
#
# ```math
# \kappa \nabla^2 b_{\rm diff} = 0 \, ,
# ```
#
# with the same boundary conditions same as our setup. In this case, we can readily find that
#
# ```math
# b_{\rm diff}(x, z) = b_s(x) \frac{\cosh \left [2 \pi (H + z) / L_x \right ]}{\cosh(2 \pi H / L_x)} \, ,
# ```
# where $b_s(x)$ is the surface boundary condition. The diffusive solution implies 
# ``\langle \chi_{\rm diff} \rangle = \kappa b_*^2 \pi \tanh(2 \pi Η /Lx) / (L_x H)``.
#
# We use the loaded `FieldTimeSeries` to compute the Nusselt number from buoyancy and the volume
# average kinetic energy of the fluid.
#
# First we compute the diffusive buoyancy dissipation, ``\chi_{\rm diff}`` (which is just a
# scalar):

χ_diff = κ * b★^2 * π * tanh(2π * H / Lx) / (Lx * H)
nothing # hide

# We recover the time from the saved `FieldTimeSeries` and construct two empty arrays to store
# the volume-averaged kinetic energy and the instantaneous Nusselt number,

t = b_timeseries.times

kinetic_energy, Nu = zeros(length(t)), zeros(length(t))
nothing # hide

# Now we can loop over the fields in the `FieldTimeSeries`, compute kinetic energy and ``Nu``,
# and plot. We make use of `Integral` to compute the volume integral of fields over our domain.

for i = 1:length(t)
    ke = Field(Integral(1/2 * s_timeseries[i]^2 / (Lx * H)))
    compute!(ke)
    kinetic_energy[i] = ke[1, 1, 1]
    
    χ = Field(Integral(χ_timeseries[i] / (Lx * H)))
    compute!(χ)

    Nu[i] = χ[1, 1, 1] / χ_diff
end

fig = Figure(resolution = (850, 450))
 
ax_KE = Axis(fig[1, 1], xlabel = "time", ylabel = L"KE $ / (b_* H)$")
lines!(ax_KE, t, kinetic_energy; linewidth = 3)

ax_Nu = Axis(fig[2, 1], xlabel = "time", ylabel = L"Nu")
lines!(ax_Nu, t, Nu; linewidth = 3)

current_figure() # hide
