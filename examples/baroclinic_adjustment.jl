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
using Random

Random.seed!(8675309) # for reproducible results

# ## Grid

# We use a three-dimensional channel that is periodic in the `x` direction:

Lx = 1000kilometers # east-west extent [m]
Ly = 1000kilometers # north-south extent [m]
Lz = 1kilometers    # depth [m]

grid = RectilinearGrid(size = (48, 48, 8),
                       x = (0, Lx),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0),
                       topology = (Periodic, Bounded, Bounded))

# ## Model

# We build a `HydrostaticFreeSurfaceModel` on a beta-plane centered at 45° South.

model = HydrostaticFreeSurfaceModel(grid;
                                    coriolis = BetaPlane(latitude = -45),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO())

# We start our simulation from rest with a baroclinically unstable buoyancy distribution.
# We use `ramp(y, Δy)`, defined below, to specify a front with width `Δy`
# and horizontal buoyancy gradient `M²`. We impose the front on top of a
# vertical buoyancy gradient `N²` and a bit of noise.

"""
    ramp(y, Δy)

Linear ramp from 0 to 1 between -Δy/2 and +Δy/2:

            y < -Δy/2 => ramp = 0
    -Δy/2 < y < +Δy/2 => ramp = y / Δy + 1/2
            y > +Δy/2 => ramp = 1
"""
ramp(y, Δy) = clamp(y / Δy + 1/2, 0, 1)

N² = 1e-5 # [s⁻²] buoyancy frequency / stratification
M² = 1e-7 # [s⁻²] horizontal buoyancy gradient

Δy = 100kilometers # width of the region of the front
Δb = Δy * M²       # buoyancy jump associated with the front
ϵb = 1e-2 * Δb     # noise amplitude

bᵢ(x, y, z) = N² * z + Δb * ramp(y, Δy) + ϵb * randn()

set!(model, b=bᵢ)

# Let's visualize the initial buoyancy distribution.

using CairoMakie
set_theme!(Theme(fontsize = 20))

## Horizontal coordinates in kilometers
x = xnodes(grid, Center()) / kilometer
y = ynodes(grid, Center()) / kilometer
z = znodes(grid, Center())

b = model.tracers.b

fig, ax, hm = heatmap(y, z, view(b, 1, :, :),
                      colormap = :deep,
                      axis = (xlabel = "y (km)",
                              ylabel = "z (m)",
                              title = "b(x=0, y, z, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "m s⁻²")

current_figure() #hide
fig

# ## Simulation
#
# Now let's build a `Simulation`.

simulation = Simulation(model, Δt=20minutes, stop_time=20days)

# We add a `TimeStepWizard` callback to adapt the simulation's time-step,

conjure_time_step_wizard!(simulation, IterationInterval(20), cfl=0.2, max_Δt=20minutes)

# Also, we add a callback to print a message about how the simulation is going,

using Printf

function print_progress(sim)
    u, v, w = sim.model.velocities
    progress = 100 * time(sim) / sim.stop_time

    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m s⁻¹, next Δt: %s\n",
            progress, iteration(sim), prettytime(sim), prettytime(sim.run_wall_time),
            maximum(abs, u), maximum(abs, v), maximum(abs, w), prettytime(sim.Δt))

    return nothing
end

add_callback!(simulation, print_progress, IterationInterval(100))

# ## Diagnostics/Output
#
# Here, we save the buoyancy ``b`` and the vertical vorticity ``ζ`` on the edges of our domain,
# as well as the zonal (``x``) averages of buoyancy and velocity.

u, v, w = model.velocities
ζ = ∂x(v) - ∂y(u)
B = Average(b, dims=1)
U = Average(u, dims=1)
V = Average(v, dims=1)

filename = "baroclinic_adjustment"
save_fields_interval = 0.5day

slicers = (east = (grid.Nx, :, :),
           north = (:, grid.Ny, :),
           bottom = (:, :, 1),
           top = (:, :, grid.Nz))

for (side, indices) in pairs(slicers)
    simulation.output_writers[side] = JLD2Writer(model, (; b, ζ);
                                                 filename = filename * "_$(side)_slice",
                                                 schedule = TimeInterval(save_fields_interval),
                                                 overwrite_files = true,
                                                 indices)
end

simulation.output_writers[:zonal] = JLD2Writer(model, (; b=B, u=U, v=V);
                                               filename = filename * "_zonal_average",
                                               schedule = TimeInterval(save_fields_interval),
                                               overwrite_files = true)

# Now we're ready to _run_.

@info "Running the simulation..."

## Fail the docs build if this simulation produces NaNs #hide
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation) #hide
run!(simulation)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)

# ## Visualization
#
# All that's left is to make a pretty movie.
# Actually, we make two visualizations here. First, we illustrate how to make a
# 3D visualization with `Makie`'s `Axis3` and `Makie.surface`. Then we make a movie in 2D.
# We use `CairoMakie` in this example, but note that `using GLMakie` is more
# convenient on a system with OpenGL, as figures will be displayed on the screen.
#
# ### Three-dimensional visualization
#
# We load the saved buoyancy output on the top, north, and east surfaces as `FieldTimeSeries`es.

slice_filenames = NamedTuple(side => filename * "_$(side)_slice.jld2" for side in keys(slicers))
zonal_average_filename = filename * "_zonal_average.jld2"

b_timeserieses = (east  = FieldTimeSeries(slice_filenames.east, "b"),
                  north = FieldTimeSeries(slice_filenames.north, "b"),
                  top   = FieldTimeSeries(slice_filenames.top, "b"))

B_timeseries = FieldTimeSeries(zonal_average_filename, "b")

times = B_timeseries.times
nothing #hide

# To draw the domain's surfaces with `surface!` we need the coordinates of each surface as matrices,
# using the coordinates `x`, `y` (in kilometers), and `z` (in meters) that we built above.

Nx, Ny, Nz = size(grid)

x_xz = repeat(x, 1, Nz)
y_xz_north = fill(y[end], Nx, Nz)
z_xz = repeat(z', Nx, 1)

x_yz_east = fill(x[end], Ny, Nz)
y_yz = repeat(y, 1, Nz)
z_yz = repeat(z', Ny, 1)

z_xy_top = fill(z[end], Nx, Ny)
nothing #hide

# Then we create a 3D axis. We use `zonal_slice_displacement` to control where the plot of the instantaneous
# zonal average flow is located.

fig = Figure(size = (1600, 800))

zonal_slice_displacement = 1.2

ax = Axis3(fig[2, 1],
           aspect=(1, 1, 1/5),
           xlabel = "x (km)", ylabel = "y (km)", zlabel = "z (m)",
           xlabeloffset = 100, ylabeloffset = 100, zlabeloffset = 100,
           limits = ((x[1], zonal_slice_displacement * x[end]), (y[1], y[end]), (z[1], z[end])),
           elevation = 0.45, azimuth = 6.8,
           xspinesvisible = false, zgridvisible = false,
           protrusions = 40, perspectiveness = 0.7)

# We use data from the final savepoint for the 3D plot.
# Note that this plot can easily be animated by using Makie's `Observable`.
# To dive into `Observable`s, check out
# [Makie.jl's Documentation](https://docs.makie.org/stable/explanations/observables).

n = length(times)

# Now let's make a 3D plot of the buoyancy and in front of it we'll use the zonally-averaged output
# to plot the instantaneous zonal-average of the buoyancy.

b_slices = (east  = interior(b_timeserieses.east[n], 1, :, :),
            north = interior(b_timeserieses.north[n], :, 1, :),
            top   = interior(b_timeserieses.top[n], :, :, 1))

B = interior(B_timeseries[n], 1, :, :)

clims = 1.1 .* extrema(b_slices.top)

kwargs = (colorrange=clims, colormap=:deep, shading=NoShading)

surface!(ax, x_yz_east, y_yz, z_yz;  color = b_slices.east, kwargs...)
surface!(ax, x_xz, y_xz_north, z_xz; color = b_slices.north, kwargs...)
surface!(ax, x, y, z_xy_top;         color = b_slices.top, kwargs...)

sf = surface!(ax, zonal_slice_displacement .* x_yz_east, y_yz, z_yz; color = B, kwargs...)

contour!(ax, y, z, B; transformation = (:yz, zonal_slice_displacement * x[end]),
         levels = 15, linewidth = 2, color = :black)

Colorbar(fig[2, 2], sf, label = "m s⁻²", height = Relative(0.4), tellheight=false)

title = "Buoyancy at t = " * prettytime(times[n])
fig[1, 1:2] = Label(fig, title; fontsize = 24, tellwidth = false, padding = (0, 0, -120, 0))

rowgap!(fig.layout, 1, Relative(-0.2))
colgap!(fig.layout, 1, Relative(-0.1))

save("baroclinic_adjustment_3d.png", fig)
nothing #hide

# ![](baroclinic_adjustment_3d.png)

# ### Two-dimensional movie
#
# We make a 2D movie that shows buoyancy ``b`` and vertical vorticity ``ζ`` at the surface,
# as well as the zonally-averaged zonal and meridional velocities ``U`` and ``V`` in the
# ``(y, z)`` plane. First we load the remaining `FieldTimeSeries` and build the coordinates
# of the nodes where ``ζ`` and ``V`` live, which are cell faces in the horizontal.

ζ_timeseries = FieldTimeSeries(slice_filenames.top, "ζ")
U_timeseries = FieldTimeSeries(zonal_average_filename, "u")
V_timeseries = FieldTimeSeries(zonal_average_filename, "v")

xᶠ = xnodes(grid, Face()) / kilometer
yᶠ = ynodes(grid, Face()) / kilometer

# Next, we set up a plot with 4 panels. The top panels are large and square, while
# the bottom panels get a reduced aspect ratio through `rowsize!`.

fig = Figure(size=(1800, 1000))

axb = Axis(fig[1, 2], xlabel="x (km)", ylabel="y (km)", aspect=1)
axζ = Axis(fig[1, 3], xlabel="x (km)", ylabel="y (km)", aspect=1, yaxisposition=:right)

axu = Axis(fig[2, 2], xlabel="y (km)", ylabel="z (m)")
axv = Axis(fig[2, 3], xlabel="y (km)", ylabel="z (m)", yaxisposition=:right)

rowsize!(fig.layout, 2, Relative(0.3))

# To prepare a plot for animation, we index the timeseries with an `Observable`,

n = Observable(1)

bₙ = @lift b_timeserieses.top[$n]
ζₙ = @lift ζ_timeseries[$n]
Uₙ = @lift U_timeseries[$n]
Vₙ = @lift V_timeseries[$n]
Bₙ = @lift B_timeseries[$n]

# and then build our plot:

hm = heatmap!(axb, x, y, bₙ, colorrange=(0, Δb), colormap=:thermal)
Colorbar(fig[1, 1], hm, flipaxis=false, label="Surface b(x, y) (m s⁻²)")

hm = heatmap!(axζ, xᶠ, yᶠ, ζₙ, colorrange=(-5e-5, 5e-5), colormap=:balance)
Colorbar(fig[1, 4], hm, label="Surface ζ(x, y) (s⁻¹)")

hm = heatmap!(axu, y, z, Uₙ; colorrange=(-5e-1, 5e-1), colormap=:balance)
Colorbar(fig[2, 1], hm, flipaxis=false, label="Zonally-averaged U(y, z) (m s⁻¹)")
contour!(axu, y, z, Bₙ; levels=15, color=:black)

hm = heatmap!(axv, yᶠ, z, Vₙ; colorrange=(-1e-1, 1e-1), colormap=:balance)
Colorbar(fig[2, 4], hm, label="Zonally-averaged V(y, z) (m s⁻¹)")
contour!(axv, y, z, Bₙ; levels=15, color=:black)
nothing #hide

# Finally, we're ready to record the movie.

frames = 1:length(times)

record(fig, filename * ".mp4", frames, framerate=8) do i
    n[] = i
end
nothing #hide

# ![](baroclinic_adjustment.mp4)
