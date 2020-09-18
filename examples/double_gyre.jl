# # Double Gyre
#
# This example simulates a double gyre following: https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html

using Oceananigans.Grids

grid = RegularCartesianGrid(size = (64, 64, 32),
                               x = (-2.5e6, 2.5e6),
                               y = (-2.5e6, 2.5e6),
                               z = (-1.8e3, 0),
                            halo = (2, 2, 2),
                        topology = (Bounded, Bounded, Bounded))

# ## Boundary conditions

using Oceananigans.BoundaryConditions

@inline wind_stress(x, y, t, parameters) = - parameters.τ * cos(2π * y / parameters.Ly)

u_bcs = UVelocityBoundaryConditions(grid,
              top = BoundaryCondition(Flux, wind_stress, parameters = (τ = 1e-4, Ly = grid.Ly)))

b_reference(y, parameters) = parameters.Δb / parameters.Ly * y

using Oceananigans.Utils

@inline buoyancy_flux(i, j, grid, clock, state, parameters) = @inbounds - parameters.μ * (state.tracers.b[i, j, grid.Nz] - b_reference(grid.yC[j], parameters))
b_bcs = TracerBoundaryConditions(grid, 
              top = BoundaryCondition(Flux, buoyancy_flux, discrete_form = true, parameters = (μ = 50 / 30day, Δb = 0.055, Ly = grid.Ly)))

using Oceananigans, Oceananigans.TurbulenceClosures, Oceananigans.Advection

closure = AnisotropicDiffusivity(νh = 5e3, νz = 1e-2, κh = 500, κz = 1e-2)

# closure = (AnisotropicDiffusivity(νh = 5e3, νz = 1e-2, κh = 500, κz = 1e-2),
           # AnisotropicBiharmonicDiffusivity(νh = 1e-3*grid.Δx^4/day, νz = 0, κh = 1e-3*grid.Δx^4/day, κz = 0))

model = IncompressibleModel(       architecture = CPU(),
                                    timestepper = :RungeKutta3, 
                                      # advection = CenteredFourthOrder(),
                                           grid = grid,
                                       coriolis = BetaPlane(latitude = 45),
                                       buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                                        closure = closure,
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

## Instantiate a JLD2OutputWriter to write fields. We will add it to the simulation before
## running it.
field_writer = JLD2OutputWriter(model, merge(model.velocities, model.tracers);
                                time_interval=7day,
                                prefix="double_gyre",
                                field_slicer=FieldSlicer(k=model.grid.Nz),
                                force=true)
                                                                 
# ## Running the simulation
#
# To run the simulation, we instantiate a `TimeStepWizard` to ensure stable time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 0.2.

wizard = TimeStepWizard(cfl = 0.50,
                         Δt = 120minute,
                 max_change = 1.1,
                     max_Δt = minimum([0.1*grid.Δz^2/closure.κz, 
                                       0.1*grid.Δx^2/closure.νx]))
                     # max_Δt = minimum([0.1*grid.Δz^2/closure[1].κz, 
                     #                   0.1*grid.Δx^4/closure[2].κx]))
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

simulation = Simulation(model, Δt=wizard, stop_time=10*365day, iteration_interval=200, progress=print_progress)
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
    u = file["timeseries/u/$iter"]
    v = file["timeseries/v/$iter"]
    w = file["timeseries/w/$iter"]
    t = file["timeseries/t/$iter"]

    ## Extract slices
    uxy = 1/2 * (u[1:end-1, :, end] .+ u[2:end, :, end])
    vxy = 1/2 * (v[:, 1:end-1, end] .+ v[:, 2:end, end])
    wxy = w[:, :, end-1]
    
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
                        
     wxy_plot = heatmap(x / 1e3, y / 1e3, wxy';
                               color = :balance,
                         aspectratio = :equal,
                               # clims = (-1e-2, 1e-2),
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
                             
    plot(wxy_plot, speed_plot, size=(1100, 500), title = ["u(t="*string(round(t/day, digits=1))*" day)" "speed"])

    iter == iterations[end] && close(file)
end

gif(anim, "double_gyre.gif", fps = 12) # hide
