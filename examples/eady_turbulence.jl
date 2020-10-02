# # Eady turbulence example
#
# In this example, we initialize a random velocity field and observe its viscous,
# turbulent decay in a two-dimensional domain. This example demonstrates:
#
#   * How to use a tuple of turbulence closures
#   * How to use biharmonic diffusivity
#   * How to create computed fields for output
#   * How to implement a background velocity and tracer distribution

# # The grid
#
# We use a three-dimensional grid with a depth of 1000 m and a 
# horizontal extent of 1000 km, appropriate for mesoscale ocean dynamics
# with characteristic scales of 50-200 km.

using Oceananigans

grid = RegularCartesianGrid(size=(64, 64, 16), x=(-5e5, 5e5), y=(-5e5, 5e5), z=(-1e3, 0))

# # Rotation
#
# The classical Eady problem is posed on an $f$-plane. We use a Coriolis parameter
# typical to mid-latitudes on Earth,

coriolis = FPlane(f=1e-4) # [s⁻¹]
                            
# # The background flow
#
# The Eady problem is non-linearized around a geostrophic basic state
# represented by the streamfunction,
#
# $ ψ(y, z) = - α f y (z + L_z) $,
#
# where $f$ is the Coriolis parameter, $α$ is the geostrophic shear
# and horizontal buoyancy gradient, and $L_z$ is the depth of the domain.
# The background buoyancy, including both the geostrophic flow component
# and a background stable stratification component, is
#
# $ B = B'(y) + N^2 z = f ∂_z ψ + N^2 z = - α f y + N^2 z$
#
# The background velocity field is
#
# $ U(z) = - ∂_y ψ = α (z + L_z) $
#
# The parameters $α$, $N$, $f$ and $L_z$ determine the Eady basic state.
# We have set the Coriolis parameter $f$ and $L_z$ above, and further
# choose $α$ and $N$,

background_parameters = ( α = 2.5e-4,       # s⁻¹, geostrophic shear
                          f = coriolis.f,   # s⁻¹, Coriolis parameter
                          N = 1e-2,         # s⁻¹, buoyancy frequency
                         Lz = grid.Lz)      # m, ocean depth

# The resulting Rossby radius of deformation is $R = N L_z / f = 1,000 \rm{m}$.
#
# With the parameters in hand, we construct the background fields $U$ and $B$

using Oceananigans.Fields: BackgroundField

## Background fields are defined via function of x, y, z, t, and optional parameters
U(x, y, z, t, p) = + p.α * (z + p.Lz)
B(x, y, z, t, p) = - p.α * p.f * y + p.N^2 * z

U_field = BackgroundField(U, parameters=background_parameters)
B_field = BackgroundField(B, parameters=background_parameters)

# # Boundary conditions
#
# These boundary conditions prescribe a linear drag at the bottom as a flux
# condition. We also fix the surface and bottom buoyancy to enforce a buoyancy
# gradient `N²`.

drag_parameters = (Cd = 2e-3, # drag coefficient
                    f = coriolis.f,
                   Lz = grid.Lz)

@inline bottom_stress_xz(i, j, grid, clock, model_fields, p) = @inbounds p.Cd * p.f * p.Lz * model_fields.u[i, j, 1]
@inline bottom_stress_yz(i, j, grid, clock, model_fields, p) = @inbounds p.Cd * p.f * p.Lz * model_fields.v[i, j, 1]

linear_drag_u = BoundaryCondition(Flux, bottom_stress_xz, discrete_form=true, parameters=drag_parameters)
linear_drag_v = BoundaryCondition(Flux, bottom_stress_yz, discrete_form=true, parameters=drag_parameters)

u_bcs = UVelocityBoundaryConditions(grid, bottom = linear_drag_u) 
v_bcs = VVelocityBoundaryConditions(grid, bottom = linear_drag_v)

# # Turbulence closures
#
# We use a horizontal biharmonic diffusivity and a Laplacian vertical diffusivity
# to dissipate energy in the Eady problem.
# To use both of these closures at the same time, we set the keyword argument
# `closure` a tuple of two closures.

κ₂z = 1e-4             # Laplacian vertical viscosity and diffusivity, [m² s⁻¹]
κ₄h = 1e-6 * grid.Δx^4 # Biharmonic horizontal viscosity and diffusivity, [m⁴ s⁻¹]

Laplacian_vertical_diffusivity = AnisotropicDiffusivity(νh=0, κh=0, νz=κ₂z, κz=κ₂z)
biharmonic_horizontal_diffusivity = AnisotropicBiharmonicDiffusivity(νh=κ₄h, κh=κ₄h)

# # Model instantiation

using Oceananigans.Advection: WENO5

model = IncompressibleModel(
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

# # Initial conditions
#
# For initial conditions we impose a linear stratifificaiton with some
# random noise.

## A noise function, damped at the top and bottom
Ξ(z) = rand() * z/grid.Lz * (z/grid.Lz + 1)

## Large amplitude noise to rapidly stimulate instability
u₀(x, y, z) = background_parameters.α * grid.Lz * 1e-1 * Ξ(z)
v₀(x, y, z) = background_parameters.α * grid.Lz * 1e-1 * Ξ(z)
w₀(x, y, z) = background_parameters.α * grid.Lz * 1e-4 * Ξ(z)

set!(model, u=u₀, v=v₀, w=w₀)

# # Simulation set-up
#
# We set up a simulation involving a preliminary integration for 10 days.
# We then stop the simulation and add a JLD2OutputWriter that saves the vertical
# velocity, vertical vorticity, and divergence every 2 iterations.
# We then run for 100 more iterations and plot the 50 frames into an animation.
#
# ## The `TimeStepWizard`
#
# The TimeStepWizard manages the time-step adaptively, keeping the CFL close to a
# desired value.

using Oceananigans.Utils: minute, hour, day

wizard = TimeStepWizard(cfl=0.5, Δt=10minute, max_change=1.1, max_Δt=1day)

# ## A progress messenger
#
# We write a function that prints out a helpful progress message while the simulation runs.

using Printf
using Oceananigans.Diagnostics: AdvectiveCFL

CFL = AdvectiveCFL(wizard)

start_time = time_ns()

progress(sim) = @printf("i: % 6d, sim time: % 10s, wall time: % 10s, Δt: % 10s, CFL: %.2e\n",
                        sim.model.clock.iteration,
                        prettytime(sim.model.clock.time),
                        prettytime(1e-9 * (time_ns() - start_time)),
                        prettytime(sim.Δt.Δt),
                        CFL(sim.model))

# ## Build the simulation
#
# We're ready to build and run the simulation. We ask for a progress message and time-step update
# every 100 iterations,

simulation = Simulation(model, Δt = wizard, iteration_interval = 100,
                                                     stop_time = 10day,
                                                      progress = progress)

# and then we spinup the Eady problem!

run!(simulation)

# ## Output and plotting
#
# With perfect confidence that the Eady problem has spun up, we prepare a
# secondary simulation for the purpose of visualizing the ensuring baroclinic turbulence.
# We'd like to plot vertical vorticity and divergence, so we create
# ComputedFields that will compute and save them during our second "diagnostic"
# simulation.

using Oceananigans.OutputWriters, Oceananigans.AbstractOperations
using Oceananigans.Fields: ComputedField

u, v, w = model.velocities # extract velocities

## ComputedFields take "AbstractOperations" on Fields as input:
ζ = ComputedField(∂x(v) - ∂y(u))
δ = ComputedField(-∂z(w))

# With the vertical vorticity, `ζ`, and the horizontal divergence, `δ` in hand,
# we create a JLD2OutputWriter that saves `w`, `ζ`, and `δ` and add it to 
# `simulation`:

simulation.output_writers[:fields] = JLD2OutputWriter(model, (w=model.velocities.w, ζ=ζ, δ=δ),
                                                                  prefix = "eady_turbulence",
                                                      iteration_interval = 2,
                                                                   force = true)

# Finally, we run `simulation` for 100 more iterations

simulation.stop_time = Inf
simulation.stop_iteration = simulation.model.clock.iteration + 200
run!(simulation)

# # Visualizing Eady turbulence
#
# We animate the results by opening the JLD2 file, extracting data for
# the iterations we ended up saving at, and ploting slices of the saved
# fields. We prepare for animating the flow by creating coordinate arrays,
# opening the file, building a vector of the iterations that we saved
# data at, and defining a function for computing colorbar limits: 

using JLD2, Plots, Printf, Oceananigans.Grids

using Oceananigans.Grids: x_domain, y_domain, z_domain # for nice domain limits

pyplot() # pyplot backend is a bit nicer than GR

## Coordinate arrays
xζ, yζ, zζ = nodes(ζ)
xδ, yδ, zδ = nodes(δ)
xw, yw, zw = nodes(w)

## Open the file with our data
file = jldopen(simulation.output_writers[:fields].filepath)

## Extract a vector of iterations
iterations = parse.(Int, keys(file["timeseries/t"]))

function nice_divergent_levels(c, clim)
    levels = range(-clim, stop=clim, length=10)

    cmax = maximum(abs, c)

    if clim < cmax # add levels on either end
        levels = vcat([-cmax], range(-clim, stop=clim, length=10), [cmax])
    end

    return levels
end

# Now we're ready to animate.

@info "Making an animation from saved data..."

anim = @animate for (i, iter) in enumerate(iterations)

    @info "Drawing frame $i from iteration $iter \n"

    ## Load 3D fields from file
    ζ = file["timeseries/ζ/$iter"][:, :, grid.Nz]
    δ = file["timeseries/δ/$iter"][:, :, grid.Nz]
    w = file["timeseries/w/$iter"][:, 1, :]

    ζlim = 0.5 * maximum(abs, ζ)
    δlim = 0.5 * maximum(abs, δ)
    wlim = 0.5 * maximum(abs, w)

    ζlevels = nice_divergent_levels(ζ, ζlim)
    δlevels = nice_divergent_levels(δ, δlim)
    wlevels = nice_divergent_levels(w, wlim)

    ζ_plot = contourf(xζ, yζ, ζ'; color = :balance,
                            aspectratio = :equal,
                                 legend = false,
                                  clims = (-ζlim, ζlim),
                                 levels = ζlevels,
                                  xlims = x_domain(grid),
                                  ylims = y_domain(grid),
                                 xlabel = "x (m)",
                                 ylabel = "y (m)")
    
    δ_plot = contourf(xδ, yδ, δ'; color = :balance,
                            aspectratio = :equal,
                                 legend = false,
                                  clims = (-δlim, δlim),
                                 levels = δlevels,
                                  xlims = x_domain(grid),
                                  ylims = y_domain(grid),
                                 xlabel = "x (m)",
                                 ylabel = "y (m)")

    w_plot = contourf(xw / 1e2, zw, w'; color = :balance,
                            aspectratio = :equal,
                                 legend = false,
                                  clims = (-wlim, wlim),
                                 levels = wlevels,
                                  xlims = x_domain(grid) ./ 1e2,
                                  ylims = z_domain(grid),
                                 xlabel = "x (m)",
                                 ylabel = "z (m)")

    plot(ζ_plot, δ_plot, w_plot, layout=(1, 3), size=(2000, 800),
         title = ["ζ(x, y, z=0, t) (1/s)" "δ(x, y, z=0, z, t) (1/s)" "w(x, y=0, z, t) (m/s)"])

    iter == iterations[end] && close(file)
end

mp4(anim, "eady_turbulence.mp4", fps = 4) # hide
