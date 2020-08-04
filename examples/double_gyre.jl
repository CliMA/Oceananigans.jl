# # Double Gyre
#
# This example simulates a double gyre following: https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html

using Oceananigans.Grids

grid = RegularCartesianGrid(size = (64, 64, 16),
                               x = (-2e6, 2e6),
                               y = (-2e6, 2e6),
                               z = (-2e3, 0),
                        topology = (Bounded, Bounded, Bounded))

# ## Boundary conditions

using Oceananigans.BoundaryConditions

wind_stress(x, y, t, parameters) = - parameters.τ * cos(2π * y / parameters.Ly)

u_bcs = UVelocityBoundaryConditions(grid,
              top = UVelocityBoundaryCondition(Flux, :z, wind_stress, (τ = 1e-4, Ly = grid.Ly)))

using Oceananigans.Forcing, Oceananigans.Utils

b_reference(y, parameters) = parameters.Δb / parameters.Ly * y

buoyancy_relaxation(i, j, k, grid, clock, state, parameters) =
    ifelse(k==grid.Nz, - parameters.μ * (state.tracers.b[i, j, k] - b_reference(grid.yC[j], parameters)), 0.0)

buoyancy_forcing = ParameterizedForcing(buoyancy_relaxation, (μ = 1 / 30day, Δb = 0.06, Ly = grid.Ly))

using Oceananigans, Oceananigans.TurbulenceClosures

model = IncompressibleModel(       architecture = CPU(),
                                           grid = grid,
                                       coriolis = BetaPlane(latitude = 45),
                                       buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                                        closure = AnisotropicDiffusivity(νh = 5e3, νz = 1e-2, κh = 500, κz = 1e-2),
                            boundary_conditions = (u=u_bcs,), 
                                        forcing = ModelForcing(b=buoyancy_forcing))
nothing # hide

## Temperature initial condition: a stable density gradient with random noise superposed.
b₀(x, y, z) = buoyancy_forcing.parameters.Δb * (1 + z / grid.Lz)

set!(model, b=b₀)

# ## Set up output
#
# We set up an output writer that saves all velocity fields, tracer fields, and the subgrid
# turbulent diffusivity associated with `model.closure`. The `prefix` keyword argument
# to `JLD2OutputWriter` indicates that output will be saved in
# `double_gyre.jld2`.

using Oceananigans.OutputWriters

## Create a NamedTuple containing all the fields to be outputted.
fields_to_output = merge(model.velocities, model.tracers)
nothing # hide

## Instantiate a JLD2OutputWriter to write fields. We will add it to the simulation before
## running it.
field_writer = JLD2OutputWriter(model, FieldOutputs(fields_to_output); interval=2hour,
                                prefix="double_gyre", force=true)

# ## Running the simulation
#
# To run the simulation, we instantiate a `TimeStepWizard` to ensure stable time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 0.2.

wizard = TimeStepWizard(cfl=0.1, Δt=5minute, max_change=1.1, max_Δt=20minute)
nothing # hide

# Finally, we set up and run the the simulation.

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

simulation = Simulation(model, Δt=wizard, stop_time=30day, progress_frequency=10, progress=print_progress)
simulation.output_writers[:fields] = field_writer

run!(simulation)


# # Making a neat movie
#
# We look at the results by plotting vertical slices of $u$ and $w$, and a horizontal
# slice of $w$ to look for Langmuir cells.

# Making the coordinate arrays takes a few lines of code,

xu, yu, zu = nodes(model.velocities.u)
xu, yu, zu = xu[:], yu[:], zu[:]
nothing # hide

# Next, we open the JLD2 file, and extract the iterations we ended up saving at,

using JLD2, Plots

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))
nothing # hide

# This utility is handy for calculating nice contour intervals:

function nice_divergent_levels(c, clim)
    levels = range(-clim, stop=clim, length=20)

    cmax = maximum(abs, c)

    if clim < cmax # add levels on either end
        levels = vcat([-cmax], range(-clim, stop=clim, length=10), [cmax])
    end

    return levels
end
nothing # hide

# Finally, we're ready to animate.

@info "Making an animation from the saved data..."

anim = @animate for (i, iter) in enumerate(iterations)

    @info "Drawing frame $i from iteration $iter \n"

    ## Load 3D fields from file, omitting halo regions
    u = file["timeseries/u/$iter"][2:end-1, 2:end-1, 2:end-1]

    ## Extract slices
    uxy = u[:, :, end]

    ulim = 1.0
    ulevels = nice_divergent_levels(u, ulim)

    uxy_plot = heatmap(xu, yu, uxy';
                              color = :balance,
                        aspectratio = :equal,
                              # clims = (-ulim, ulim),
                             # levels = ulevels,
                             xlabel = "x (m)",
                             ylabel = "y (m)")
                        
    # plot(uxy_plot, size=(500, 500), title = ["u(x, y, z=0, t) (m/s)"])

    iter == iterations[end] && close(file)
end

gif(anim, "double_gyre.gif", fps = 8) # hide
