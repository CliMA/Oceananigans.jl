# # Baroclinic adjustment
#
# In this example, we simulate the evolution and equilibration of a baroclinically
# unstable front.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie"
# ```

using Oceananigans
using Oceananigans.Units

# ## Grid

# We use a three-dimensional channel that is periodic in the `x` direction:

Lx = 1000kilometers # east-west extent [m]
Ly = 1000kilometers # north-south extent [m]
Lz = 1kilometers    # depth [m]

Nx = 64
Ny = 64
Nz = 40

grid = RectilinearGrid(size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0),
                       topology = (Periodic, Bounded, Bounded))

# ## Model

# We built a `HydrostaticFreeSurfaceModel` with an `ImplicitFreeSurface` solver.
# Regarding Coriolis, we use a beta-plane centered at 45° South.

model = HydrostaticFreeSurfaceModel(; grid,
                                    coriolis = BetaPlane(latitude = -45),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO())

# We want to initialize our model with a baroclinically unstable front plus some small-amplitude
# noise.

"""
    ramp(y, Δy)

Linear ramp from 0 to 1 between -Δy/2 and +Δy/2.

For example:
```
            y < -Δy/2 => ramp = 0
    -Δy/2 < y < -Δy/2 => ramp = y / Δy
            y >  Δy/2 => ramp = 1
```
"""
ramp(y, Δy) = min(max(0, y/Δy + 1/2), 1)
nothing #hide

# We then use `ramp(y, Δy)` to construct an initial buoyancy configuration of a baroclinically
# unstable front. The front has a buoyancy jump `Δb` over a latitudinal width `Δy`.

N² = 4e-6 # [s⁻²] buoyancy frequency / stratification
M² = 8e-8 # [s⁻²] horizontal buoyancy gradient

Δy = 50kilometers # width of the region of the front
Δb = Δy * M²      # buoyancy jump associated with the front
ϵb = 1e-2 * Δb    # noise amplitude

bᵢ(x, y, z) = N² * z + Δb * ramp(y, Δy) + ϵb * randn()

set!(model, b=bᵢ)

# Let's visualize the initial buoyancy distribution.

using CairoMakie

x, y, z = 1e-3 .* nodes((Center, Center, Center), grid) # convert m -> km

b = model.tracers.b

fig, ax, hm = heatmap(y, z, interior(b)[1, :, :],
                      colormap=:deep,
                      axis = (xlabel = "y [km]",
                              ylabel = "z [km]",
                              title = "b(x=0, y, z, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m s⁻²]")

save("initial_buoyancy.svg", fig); nothing # hide

# ![](initial_buoyancy.svg)

# Now let's built a `Simulation`.

Δt₀ = 5minutes
stop_time = 40days

simulation = Simulation(model, Δt=Δt₀, stop_time=stop_time)

# We add a `TimeStepWizard` callback to adapt the simulation's time-step,

wizard = TimeStepWizard(cfl=0.2, max_change=1.1, max_Δt=20minutes)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

# Also, we add a callback to print a message about how the simulation is going,

using Printf

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()
    
    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

# ## Diagnostics/Output

# Add some diagnostics. Here, we save the buoyancy, ``b``, at the edges of our domain as well as
# the zonal (``x``) average of buoyancy.

u, v, w = model.velocities

B = Field(Average(b, dims=1))

filename = "baroclinic_adjustment"
save_fields_interval = 0.5day

slicers = (west = (1, :, :),
           east = (grid.Nx, :, :),
           south = (:, 1, :),
           north = (:, grid.Ny, :),
           bottom = (:, :, 1),
           top = (:, :, grid.Nz))

for side in keys(slicers)
    indices = slicers[side]

    simulation.output_writers[side] = JLD2OutputWriter(model, (; b);
                                                       filename = filename * "_$(side)_slice",
                                                       schedule = TimeInterval(save_fields_interval),
                                                       overwrite_existing = true,
                                                       indices)
end

simulation.output_writers[:zonal] = JLD2OutputWriter(model, (b=B,);
                                                     filename = filename * "_zonal_average",
                                                     schedule = TimeInterval(save_fields_interval),
                                                     overwrite_existing = true)

# Now let's run!

@info "Running the simulation..."

run!(simulation)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)

# ## Visualization

# Now we are ready to visualize our resutls! We use `CairoMakie` in this example.
# On a system with OpenGL `using GLMakie` is more convenient as figures will be
# displayed on the screen.

using CairoMakie

# We load the saved buoyancy output on the top, bottom, and east surface as `FieldTimeSeries`es.

filename = "baroclinic_adjustment"

sides = keys(slicers)

slice_filenames = NamedTuple(side => filename * "_$(side)_slice.jld2" for side in sides)

b_timeserieses = (east   = FieldTimeSeries(slice_filenames.east, "b"),
                  north  = FieldTimeSeries(slice_filenames.north, "b"),
                  bottom = FieldTimeSeries(slice_filenames.bottom, "b"),
                  top    = FieldTimeSeries(slice_filenames.top, "b"))

avg_b_timeseries = FieldTimeSeries(filename * "_zonal_average.jld2", "b")

nothing #hide

# We build the coordinates. We rescale horizontal coordinates so that they correspond to kilometers.

x, y, z = nodes(b_timeserieses.east)

x = x .* 1e-3 # convert m -> km
y = y .* 1e-3 # convert m -> km

x_xz = repeat(x, 1, Nz)
y_xz_north = y[end] * ones(Nx, Nz)
z_xz = repeat(reshape(z, 1, Nz), Nx, 1)

x_yz_east = x[end] * ones(Ny, Nz)
y_yz = repeat(y, 1, Nz)
z_yz = repeat(reshape(z, 1, Nz), grid.Ny, 1)

x_xy = x
y_xy = y
z_xy_top = z[end] * ones(grid.Nx, grid.Ny)
z_xy_bottom = z[1] * ones(grid.Nx, grid.Ny)
nothing #hide

# Then we create a 3D axis. We use `zonal_slice_displacement` to control where the plot of the instantaneous
# zonal average flow is located.

fig = Figure(resolution = (900, 520))

zonal_slice_displacement = 1.2

ax = Axis3(fig[2, 1], aspect=(1, 1, 1/5),
           xlabel="x (km)", ylabel="y (km)", zlabel="z (m)",
           limits = ((x[1], zonal_slice_displacement * x[end]), (y[1], y[end]), (z[1], z[end])),
           elevation = 0.45, azimuth = 6.8,
           xspinesvisible = false, zgridvisible=false,
           protrusions=40,
           perspectiveness=0.7)

nothing #hide

# We use Makie's `Observable` to animate the data. To dive into how `Observable`s work we
# refer to [Makie.jl's Documentation](https://makie.juliaplots.org/stable/documentation/nodes/index.html).

n = Observable(1)

# Now let's make a 3D plot of the buoyancy and in front of it we'll use the zonally-averaged output
# to plot the instantaneous zonal-average of the buoyancy.

b_slices = (east   = @lift(interior(b_timeserieses.east[$n], 1, :, :)),
            north  = @lift(interior(b_timeserieses.north[$n], :, 1, :)),
            bottom = @lift(interior(b_timeserieses.bottom[$n], :, :, 1)),
            top    = @lift(interior(b_timeserieses.top[$n], :, :, 1)))

avg_b = @lift interior(avg_b_timeseries[$n], 1, :, :)

clims = @lift 1.1 .* extrema(b_timeserieses.top[$n][:])

kwargs = (colorrange = clims, colormap = :deep)

surface!(ax, x_yz_east, y_yz, z_yz;    color = b_slices.east, kwargs...)
surface!(ax, x_xz, y_xz_north, z_xz;   color = b_slices.north, kwargs...)
surface!(ax, x_xy, y_xy, z_xy_bottom ; color = b_slices.bottom, kwargs...)
surface!(ax, x_xy, y_xy, z_xy_top;     color = b_slices.top, kwargs...)

sf = surface!(ax, zonal_slice_displacement .* x_yz_east, y_yz, z_yz; color = avg_b, kwargs...)

contour!(ax, y, z, avg_b; transformation = (:yz, zonal_slice_displacement * x[end]),
         levels = 15, linewidth = 2, color = :black)

Colorbar(fig[2, 2], sf, label = "m s⁻²", height = 200, tellheight=false)

# Finally, we add a figure title with the time of the snapshot and then record a movie.

times = avg_b_timeseries.times

title = @lift "Buoyancy at t = " * string(round(times[$n] / day, digits=1)) * " days"

fig[1, 1:2] = Label(fig, title; textsize = 24, tellwidth = false, padding = (0, 0, -120, 0))

frames = 1:length(times)

record(fig, filename * ".mp4", frames, framerate=8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end
nothing #hide

# ![](baroclinic_adjustment.mp4)
