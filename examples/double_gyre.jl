# # Double Gyre
#
# This example simulates a double gyre following: https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html

using Oceananigans.Grids

grid = RegularCartesianGrid(size = (128, 128, 32),
                               x = (-2e6, 2e6),
                               y = (-2e6, 2e6),
                               z = (-1e3, 0),
                        topology = (Bounded, Bounded, Bounded))

# ## Boundary conditions

using Oceananigans.BoundaryConditions

@inline wind_stress(x, y, t, parameters) = - parameters.τ * cos(2π * y / parameters.Ly)

u_bcs = UVelocityBoundaryConditions(grid,
              top = UVelocityBoundaryCondition(Flux, :z, wind_stress, (τ = 1e-4, Ly = grid.Ly)))

b_reference(y, parameters) = parameters.Δb / parameters.Ly * y

@inline buoyancy_flux(i, j, grid, clock, state, parameters) = @inbounds - parameters.μ * (state.tracers.b[i, j, grid.Nz] - b_reference(grid.yC[j], parameters))

using Oceananigans.Utils

b_bcs = TracerBoundaryConditions(grid,
              top = ParameterizedBoundaryCondition(Flux, buoyancy_flux, (μ = 50 / 30day, Δb = 0.06, Ly = grid.Ly)))

using Oceananigans, Oceananigans.TurbulenceClosures

model = IncompressibleModel(       architecture = CPU(),
                                           grid = grid,
                                       coriolis = BetaPlane(latitude = 45),
                                       buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                                        closure = AnisotropicDiffusivity(νh = 5e3, νz = 1e-2, κh = 500, κz = 1e-2),
                            boundary_conditions = (u=u_bcs, b=b_bcs))
nothing # hide

## Temperature initial condition: a stable density gradient with random noise superposed.
b₀(x, y, z) = b_bcs.z.top.condition.parameters.Δb * (1 + z / grid.Lz)

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
field_writer = JLD2OutputWriter(model, FieldOutputs(fields_to_output); interval=7day,
                                prefix="double_gyre", force=true)

# ## Running the simulation
#
# To run the simulation, we instantiate a `TimeStepWizard` to ensure stable time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 0.2.

wizard = TimeStepWizard(cfl = 0.15, Δt = 15minute, max_change = 1.1, max_Δt = 0.02*grid.Δz^2/model.closure.κz.b)
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

simulation = Simulation(model, Δt=wizard, stop_time=2*730day, progress_frequency=100, progress=print_progress)
simulation.output_writers[:fields] = field_writer

run!(simulation)


# # Making a neat movie
#
# We look at the results by plotting vertical slices of $u$ and $w$, and a horizontal
# slice of $w$ to look for Langmuir cells.

# Making the coordinate arrays takes a few lines of code,

x, y, z = nodes(model.tracers.b)
x, y, z = x[:], y[:], z[:]
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
    v = file["timeseries/v/$iter"][2:end-1, 2:end-1, 2:end-1]
    t = file["timeseries/t/$iter"]

    ## Extract slices
    uxy = 1/2 * (u[1:end-1, :, end] .+ u[2:end, :, end])
    vxy = 1/2 * (v[:, 1:end-1, end] .+ v[:, 2:end, end])
    
    speed = @. sqrt(uxy^2 + vxy^2)
    
    ulim = 1.0
    ulevels = nice_divergent_levels(u, ulim)

    uxy_plot = heatmap(x / 1e3, y / 1e3, uxy';
                              color = :balance,
                        aspectratio = :equal,
                              clims = (-2, 2),
                             # levels = ulevels,
                              xlims = (-grid.Lx/2e3, grid.Lx/2e3),
                              ylims = (-grid.Ly/2e3, grid.Ly/2e3),
                             xlabel = "x (km)",
                             ylabel = "y (km)")
                        
    speed_plot = heatmap(x / 1e3, y / 1e3 , speed';
                              color = :deep,
                        aspectratio = :equal,
                              clims = (0, 2.0),
                             # levels = ulevels,
                              xlims = (-grid.Lx/2e3, grid.Lx/2e3),
                              ylims = (-grid.Ly/2e3, grid.Ly/2e3),
                             xlabel = "x (km)",
                             ylabel = "y (km)")
                             
    plot(uxy_plot, speed_plot, size=(1100, 500), title = ["u(t="*string(round(t/day, digits=1))*" day)" "speed"])

    iter == iterations[end] && close(file)
end

gif(anim, "double_gyre.gif", fps = 12) # hide
