# # Langmuir turbulence example
#
# This example implements the Langmuir turbulence simulation reported in section
# 4 of
#
# > [McWilliams, J. C. et al., "Langmuir Turbulence in the ocean," Journal of Fluid Mechanics (1997)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/langmuir-turbulence-in-the-ocean/638FD0E368140E5972144348DB930A38).
#
# This example demonstrates 
#
#   * How to run large eddy simulations with surface wave effects via the
#     Craik-Leibovich approximation
#
#   * How to specify time-averaged output

using Oceananigans

# ## Model set-up
#
# To build the model, we specify the grid, Stokes drift, boundary conditions, and
# Coriolis parameter.
#
# ### Domain and numerical grid specification
#
# We create a grid with modest resolution. The grid extent is similar, but not
# exactly the same as that in McWilliams et al. (1997).

grid = RegularCartesianGrid(size=(32, 32, 48), extent=(128, 128, 96))

# ### The Stokes Drift profile
#
# The surface wave Stokes drift profile prescribed in McWilliams et al. (1997)
# corresponds to a 'monochromatic' (that is, single-frequency) wave field.
#
# A monochromatic wave field is characterized by its wavelength and amplitude
# (half the distance from wave crest to wave trough), which determine the wave
# frequency and the vertical scale of the Stokes drift profile.

using Oceananigans.Buoyancy: g_Earth

 amplitude = 0.8 # m
wavelength = 60 # m
wavenumber = 2π / wavelength # m⁻¹
 frequency = sqrt(g_Earth * wavenumber) # s⁻¹

## The vertical scale over which the Stokes drift of a monochromatic surface wave
## decays away from the surface is `1/2wavenumber`, or
const vertical_scale = wavelength / 4π

## Stokes drift velocity at the surface
const Uˢ = amplitude^2 * wavenumber * frequency # m s⁻¹

# The `const` declarations ensure that Stokes drift functions compile on the GPU.
# To run this example on the GPU, write `architecture = GPU()` in the constructor
# for `IncompressibleModel` below.
#
# The Stokes drift profile is

uˢ(z) = Uˢ * exp(z / vertical_scale)

# which we'll need for the initial condition.
#
# Note that `Oceananigans.jl` implements the Lagrangian-mean form of the Craik-Leibovich
# equations. This means `Oceananigans.jl` takes the *vertical derivative of the Stokes drift*
# as input, rather than the Stokes drift profile itself.
#
# The vertical derivative of the Stokes drift is

∂z_uˢ(z, t) = 1 / vertical_scale * Uˢ * exp(z / vertical_scale)

# Finally, we note that the time-derivative of the Stokes drift must be provided
# if the Stokes drift changes in time. In this example, the Stokes drift is constant
# and thus the time-derivative of the Stokes drift is 0.

# ### Boundary conditions
#
# At the surface at ``z=0``, McWilliams et al. (1997) impose a wind stress
# on ``u``,

using Oceananigans.BoundaryConditions

Qᵘ = -3.72e-5 # m² s⁻², surface kinematic momentum flux

u_boundary_conditions = UVelocityBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᵘ))

# On buoyancy, the initial condition and bottom boundary condition impose the
# linear buoyancy gradient `N²`. McWilliams et al. (1997) also impose a weak,
# destabilizing flux of buoyancy at the surface to avoid spurious laminarization of the
# near-surface velocity field.

Qᵇ = 2.307e-9 # m³ s⁻², surface buoyancy flux
N² = 1.936e-5 # s⁻², initial and bottom buoyancy gradient

b_boundary_conditions = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᵇ),
                                                       bottom = BoundaryCondition(Gradient, N²))

# Note that Oceananigans uses "positive upward" conventions for all fluxes. In consequence,
# a negative flux at the surface drives positive velocities, and a positive flux of
# buoyancy drives cooling.

# ### Coriolis parameter
#
# McWilliams et al. (1997) use

coriolis = FPlane(f=1e-4) # s⁻¹

# which is typical for mid-latitudes on Earth.

# ## Model instantiation
#
# Finally, we are ready to build the model. We use the `AnisotropicMinimumDissipation`
# model for large eddy simulation. Because our Stokes drift does not vary in ``x, y``,
# we use `UniformStokesDrift`, which expects Stokes drift functions of ``z, t`` only.

using Oceananigans.Advection
using Oceananigans.Buoyancy: BuoyancyTracer
using Oceananigans.SurfaceWaves: UniformStokesDrift

model = IncompressibleModel(
           architecture = CPU(),
              advection = UpwindBiasedFifthOrder(),
            timestepper = :RungeKutta3,
                   grid = grid,
                tracers = :b,
               buoyancy = BuoyancyTracer(),
               coriolis = coriolis,
                closure = AnisotropicMinimumDissipation(),
          surface_waves = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
    boundary_conditions = (u=u_boundary_conditions, b=b_boundary_conditions),
)

# ## Initial conditions
#
# We make use of random noise concentrated in the upper 4 meters
# for buoyancy and velocity initial conditions,

Ξ(z) = randn() * exp(z / 4)
nothing # hide

# Our initial condition for buoyancy consists of a linear stratification, plus noise,

bᵢ(x, y, z) = N² * z + 1e-1 * Ξ(z) * N² * model.grid.Lz
nothing # hide

# The velocity initial condition in McWilliams et al. (1997) is zero *Eulerian-mean* velocity.
# This means that we must add the Stokes drift profile to the ``u`` velocity field.
# We also add noise scaled by the friction velocity to ``u`` and ``w``.

uᵢ(x, y, z) = uˢ(z) + sqrt(abs(Qᵘ)) * 1e-1 * Ξ(z)

wᵢ(x, y, z) = sqrt(abs(Qᵘ)) * 1e-1 * Ξ(z)

set!(model, u=uᵢ, w=wᵢ, b=bᵢ)

# ## Setting up the simulation
#
# We use the `TimeStepWizard` for adaptive time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 1.0,

using Oceananigans.Utils

wizard = TimeStepWizard(cfl=1.0, Δt=45.0, max_change=1.1, max_Δt=1minute)

# ### Nice progress messaging
#
# We define a function that prints a helpful message with
# maximum absolute value of ``u, v, w`` and the current wall clock time.

using Oceananigans.Diagnostics, Printf

umax = FieldMaximum(abs, model.velocities.u)
vmax = FieldMaximum(abs, model.velocities.v)
wmax = FieldMaximum(abs, model.velocities.w)

wall_clock = time_ns()

function print_progress(simulation)
    model = simulation.model

    ## Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   model.clock.iteration,
                   prettytime(model.clock.time),
                   prettytime(wizard.Δt),
                   umax(), vmax(), wmax(),
                   prettytime(1e-9 * (time_ns() - wall_clock))
                  )

    @info msg

    return nothing
end

# Now we create the simulation,

simulation = Simulation(model, iteration_interval = 10,
                                               Δt = wizard,
                                        stop_time = 4hours,
                                         progress = print_progress)

# ## Output
#
# ### A field writer
#
# We set up an output writer for the simulation that saves all velocity fields,
# tracer fields, and the subgrid turbulent diffusivity.

using Oceananigans.OutputWriters

output_interval = 10minutes

fields_to_output = merge(model.velocities, model.tracers, (νₑ=model.diffusivities.νₑ,))

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, fields_to_output,
                     schedule = TimeInterval(output_interval),
                     prefix = "langmuir_turbulence_fields",
                     force = true)

# ### An "averages" writer
#
# We also set up output of time- and horizontally-averaged velocity field and
# momentum fluxes

using Oceananigans.Fields

u, v, w = model.velocities

U = AveragedField(u, dims=(1, 2))
V = AveragedField(v, dims=(1, 2))
B = AveragedField(model.tracers.b, dims=(1, 2))

wu = AveragedField(w * u, dims=(1, 2))
wv = AveragedField(w * v, dims=(1, 2))

simulation.output_writers[:averages] =
    JLD2OutputWriter(model, (u=U, v=V, b=B, wu=wu, wv=wv),
                     schedule = AveragedTimeInterval(output_interval, window=5minutes),
                     prefix = "langmuir_turbulence_averages",
                     force = true)

# ## Running the simulation
#
# This part is easy,

run!(simulation)

# # Making a neat movie
#
# We look at the results by plotting vertical slices of ``u`` and ``w``, and a horizontal
# slice of ``w`` to look for Langmuir cells.

k = searchsortedfirst(grid.zF[:], -8)
nothing # hide

# Making the coordinate arrays takes a few lines of code,

using Oceananigans.Grids

xw, yw, zw = nodes(model.velocities.w)
xu, yu, zu = nodes(model.velocities.u)
nothing # hide

# Next, we open the JLD2 file, and extract the iterations we ended up saving at,

using JLD2, Plots

fields_file = jldopen(simulation.output_writers[:fields].filepath)
averages_file = jldopen(simulation.output_writers[:averages].filepath)

iterations = parse.(Int, keys(fields_file["timeseries/t"]))

# This utility is handy for calculating nice contour intervals:

function nice_divergent_levels(c, clim; nlevels=21)
    levels = range(-clim, stop=clim, length=nlevels)

    cmax = maximum(abs, c)

    if clim < cmax # add levels on either end
        levels = vcat([-cmax], levels, [cmax])
    end

    return levels
end
nothing # hide

# Finally, we're ready to animate.

@info "Making an animation from the saved data..."

anim = @animate for (i, iter) in enumerate(iterations)

    @info "Drawing frame $i from iteration $iter \n"

    ## Load 3D fields from fields_file
    w_snapshot = fields_file["timeseries/w/$iter"]
    u_snapshot = fields_file["timeseries/u/$iter"]

    B_snapshot = averages_file["timeseries/b/$iter"][1, 1, :]
    U_snapshot = averages_file["timeseries/u/$iter"][1, 1, :]
    V_snapshot = averages_file["timeseries/v/$iter"][1, 1, :]
    wu_snapshot = averages_file["timeseries/wu/$iter"][1, 1, :]
    wv_snapshot = averages_file["timeseries/wu/$iter"][1, 1, :]

    ## Extract slices
    wxy = w_snapshot[:, :, k]
    wxz = w_snapshot[:, 1, :]
    uxz = u_snapshot[:, 1, :]

    wlim = 0.02
    ulim = 0.05
    wlevels = nice_divergent_levels(w, wlim)
    ulevels = nice_divergent_levels(w, ulim)

    B_plot = plot(B_snapshot, zu,
                  label = nothing,
                  legend = :bottom,
                  xlabel = "Buoyancy",
                  ylabel = "z (m)")

    U_plot = plot([U_snapshot V_snapshot], zu,
                  label = ["\$ \\bar u \$" "\$ \\bar v \$"],
                  legend = :bottom,
                  xlabel = "Velocities",
                  ylabel = "z (m)")

    wu_label = "\$ \\overline{wu} \$"
    wv_label = "\$ \\overline{wv} \$"

    fluxes_plot = plot([wu_snapshot, wv_snapshot], zw,
                       label = [wu_label wv_label],
                       legend = :bottom,
                       xlabel = "Momentum fluxes",
                       ylabel = "z (m)")

    wxy_plot = contourf(xw, yw, wxy';
                              color = :balance,
                          linewidth = 0,
                        aspectratio = :equal,
                              clims = (-wlim, wlim),
                             levels = wlevels,
                              xlims = (0, grid.Lx),
                              ylims = (0, grid.Ly),
                             xlabel = "x (m)",
                             ylabel = "y (m)")

    wxz_plot = contourf(xw, zw, wxz';
                              color = :balance,
                          linewidth = 0,
                        aspectratio = :equal,
                              clims = (-wlim, wlim),
                             levels = wlevels,
                              xlims = (0, grid.Lx),
                              ylims = (-grid.Lz, 0),
                             xlabel = "x (m)",
                             ylabel = "z (m)")

    uxz_plot = contourf(xu, zu, uxz';
                              color = :balance,
                          linewidth = 0,
                        aspectratio = :equal,
                              clims = (-ulim, ulim),
                             levels = ulevels,
                              xlims = (0, grid.Lx),
                              ylims = (-grid.Lz, 0),
                             xlabel = "x (m)",
                             ylabel = "z (m)")

       wxy_title = "w(x, y, z=-8, t) (m s⁻¹)"
       wxz_title = "w(x, y=0, z, t) (m s⁻¹)"
       uxz_title = "u(x, y=0, z, t) (m s⁻¹)"
         B_title = "Averaged buoyancy (m² s⁻³)"
         U_title = "Averaged velocities (m s⁻¹)"
    fluxes_title = "Averaged fluxes(m² s⁻²)"
         
    plot(wxy_plot, B_plot, wxz_plot, U_plot, uxz_plot, fluxes_plot,
         layout=(3, 2), size=(1000, 1000),
         title = [wxy_title B_title wxz_title U_title uxz_title fluxes_title])

    if iter == iterations[end]
        close(fields_file)
        close(averages_file)
    end
end

gif(anim, "langmuir_turbulence.gif", fps = 8) # hide
