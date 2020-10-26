# # Langmuir turbulence example
#
# This example implements the Langmuir turbulence simulation reported in section
# 4 of
#
# [McWilliams, J. C. et al., "Langmuir Turbulence in the ocean," Journal of Fluid Mechanics (1997)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/langmuir-turbulence-in-the-ocean/638FD0E368140E5972144348DB930A38).
#
# This example demonstrates:
#
#   * how to run large eddy simulations with surface wave effects
#     via the Craik-Leibovich approximation

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

using Oceananigans.Grids

grid = RegularCartesianGrid(size=(32, 32, 48), extent=(128, 128, 96))

# ### The Stokes Drift profile
#
# The surface wave Stokes drift profile used in McWilliams et al. (1997)
# corresponds to a 'monochromatic' (that is, single-frequency) wave field with

const wavenumber = 2π / 60 # m⁻¹
nothing # hide

# and

const amplitude = 0.8 # m
nothing # hide

# The `const` declarations ensure that Stokes drift functions compile on the GPU.
# To run this example on the GPU, write `architecture = GPU()` in the constructor
# for `IncompressibleModel` below.
#
# The Stokes drift at the surface for a monochromatic, deep water wave is

using Oceananigans.Buoyancy: g_Earth

const Uˢ = amplitude^2 * wavenumber * sqrt(g_Earth * wavenumber) # m s⁻¹

# The Stokes drift profile is then,

uˢ(z) = Uˢ * exp(2wavenumber * z)
nothing # hide

# which we need for the initial condition.
#
# Note that `Oceananigans.jl` implements the Lagrangian-mean form of the Craik-Leibovich
# equations. This means `Oceananigans.jl` takes the *vertical derivative of the Stokes drift*
# as input, rather than the Stokes drift profile itself.
#
# The vertical derivative of the Stokes drift is

∂z_uˢ(z, t) = 2wavenumber * Uˢ * exp(2wavenumber * z)
nothing # hide

# Finally, we note that the time-derivative of the Stokes drift must be provided
# if the Stokes drift changes in time. In this example, the Stokes drift is constant
# and thus the time-derivative of the Stokes drift is 0.

# ### Boundary conditions
#
# At the surface at ``z=0``, McWilliams et al. (1997) impose wind stress,

Qᵘ = -3.72e-5 # m² s⁻²
nothing # hide

# and weak cooling with buoyancy flux

Qᵇ = 2.307e-9 # m³ s⁻²
nothing # hide

# Oceananigans uses "positive upward" conventions for all fluxes. In consequence,
# a negative flux at the surface drives positive velocities, and a positive flux of
# buoyancy drives cooling.
#
# The initial condition and bottom boundary condition for buoyancy
# impose a linear stratification with buoyancy frequency

N² = 1.936e-5 # s⁻²
nothing # hide

# To summarize, we impose a surface flux on ``u``,

using Oceananigans.BoundaryConditions

u_boundary_conditions = UVelocityBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᵘ))
nothing # hide

# and a surface flux and bottom linear gradient on buoyancy, ``b``,

b_boundary_conditions = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᵇ),
                                                       bottom = BoundaryCondition(Gradient, N²))
nothing # hide

# ### Coriolis parameter
#
# McWilliams et al. (1997) use

f = 1e-4 # s⁻¹
nothing # hide

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
               coriolis = FPlane(f=f),
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

using Oceananigans.Utils: minute

wizard = TimeStepWizard(cfl=1.0, Δt=45.0, max_change=1.1, max_Δt=1minute)

# ### Nice progress messaging
#
# We define a function that prints a helpful message with
# maximum absolute value of ``u, v, w`` and the current wall clock time.

using Oceananigans.Diagnostics, Printf
using Oceananigans.Utils: prettytime

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

using Oceananigans.Utils: hour # correpsonds to "1 hour", in units of seconds

simulation = Simulation(model, iteration_interval = 100,
                                               Δt = wizard,
                                        stop_time = 6hour,
                                         progress = print_progress)

# ## Output
#
# ### A field writer
#
# We set up an output writer for the simulation that saves all velocity fields,
# tracer fields, and the subgrid turbulent diffusivity.

using Oceananigans.OutputWriters

output_interval = 10minute

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
                     schedule = AveragedTimeInterval(output_interval, window=5minute),
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

xw, yw, zw = nodes(model.velocities.w)
xu, yu, zu = nodes(model.velocities.u)
nothing # hide

# Next, we open the JLD2 file, and extract the iterations we ended up saving at,

using JLD2, Plots

fields_file = jldopen(simulation.output_writers[:fields].filepath)
averages_file = jldopen(simulation.output_writers[:averages].filepath)

iterations = parse.(Int, keys(fields_file["timeseries/t"]))

# This utility is handy for calculating nice contour intervals:

function nice_divergent_levels(c, clim; nlevels=31)
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

    local w
    local u
    local B
    local U
    local V
    local wu
    local wv

    @info "Drawing frame $i from iteration $iter \n"

    ## Load 3D fields from fields_file
    w = fields_file["timeseries/w/$iter"]
    u = fields_file["timeseries/u/$iter"]

    B = averages_file["timeseries/b/$iter"][1, 1, :]
    U = averages_file["timeseries/u/$iter"][1, 1, :]
    V = averages_file["timeseries/v/$iter"][1, 1, :]
    wu = averages_file["timeseries/wu/$iter"][1, 1, :]
    wv = averages_file["timeseries/wu/$iter"][1, 1, :]

    ## Extract slices
    wxy = w[:, :, k]
    wxz = w[:, 1, :]
    uxz = u[:, 1, :]

    wlim = 0.02
    ulim = 0.05
    wlevels = nice_divergent_levels(w, wlim)
    ulevels = nice_divergent_levels(w, ulim)

    B_plot = plot(B, zu,
                  label = nothing,
                  legend = :bottom,
                  xlabel = "Buoyancy",
                  ylabel = "z (m)")

    U_plot = plot([U V], zu,
                  label = ["\$ \\bar u \$" "\$ \\bar v \$"],
                  legend = :bottom,
                  xlabel = "Velocities",
                  ylabel = "z (m)")

    wu_label = "\$ \\overline{wu} \$"
    wv_label = "\$ \\overline{wv} \$"

    fluxes_plot = plot([wu, wv], zw,
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

mp4(anim, "langmuir_turbulence.mp4", fps = 8) # hide
