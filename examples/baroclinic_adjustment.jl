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
# pkg"add Oceananigans, CairoMakie, JLD2"
# ```

using Oceananigans
using Oceananigans.Units

using Printf
using Statistics
using JLD2

# ## Grid
#
# We use a three-dimensional channel that is periodic in the `x` direction:

Lx = 1000kilometers # east-west extent [m]
Ly = 1000kilometers # north-south extent [m]
Lz = 1kilometers    # depth [m]

Nx = 64
Ny = 64
Nz = 40

grid = RectilinearGrid(CPU();
                       topology = (Periodic, Bounded, Bounded), 
                       size = (Nx, Ny, Nz), 
                       x = (0, Lx),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0),
                       halo = (3, 3, 3))

# ## Turbulence closures

# We prescribe the values of vertical viscocity and diffusivity according to the ratio
# of the vertical and lateral grid spacing.

Δx, Δz = Lx/Nx, Lz/Nz

𝒜 = Δz/Δx # Grid cell aspect ratio.

κh = 0.1    # [m² s⁻¹] horizontal diffusivity
νh = 0.1    # [m² s⁻¹] horizontal viscosity
κz = 𝒜 * κh # [m² s⁻¹] vertical diffusivity
νz = 𝒜 * νh # [m² s⁻¹] vertical viscosity

horizontal_diffusive_closure = HorizontalScalarDiffusivity(ν = νh, κ = κh)

vertical_diffusive_closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization();
                                                       ν = νz, κ = κz)

# ## Model

# We built a `HydrostaticFreeSurfaceModel` with an `ImplicitFreeSurface` solver.
# Regarding Coriolis, we use a beta-plane centered at 45° South.

model = HydrostaticFreeSurfaceModel(; grid,
                                      coriolis = BetaPlane(latitude = -45),
                                      buoyancy = BuoyancyTracer(),
                                      tracers = :b,
                                      closure = (vertical_diffusive_closure, horizontal_diffusive_closure),
                                      momentum_advection = WENO5(),
                                      tracer_advection = WENO5(),
                                      free_surface = ImplicitFreeSurface())

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

y, z = grid.yᵃᶜᵃ[1:grid.Ny], grid.zᵃᵃᶜ[1:grid.Nz]

fig, ax, hm = heatmap(y * 1e-3, z * 1e-3, interior(model.tracers.b)[1, :, :],
                      colormap=:deep,
                      axis = (xlabel = "y [km]", ylabel = "z [km]"))

Colorbar(fig[1, 2], hm)

save("initial_buoyancy.svg", fig)

# Now let's built a `Simulation`.

Δt₀ = 5minutes

simulation = Simulation(model, Δt=Δt₀, stop_time=stop_time)

# We add a `TimeStepWizard` callback to adapt the siulation's time-step,

wizard = TimeStepWizard(cfl=0.2, max_change=1.1, max_Δt=20minutes)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

# Also, we add a callback to print a message about how the simulation is going,

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

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(20))

# ## Diagnostics/Output

# Add some diagnostics. Here, we save the buoyancy, ``b``, at the edges of our domain as well as
# the zonal (``x``) averages of buoyancy and zonal velocity ``u``.

u, v, w = model.velocities
b = model.tracers.b

B = Field(Average(b, dims=1))
U = Field(Average(u, dims=1))

filename = "baroclinic_adjustment"
save_fields_interval = 0.5day
stop_time = 40days

slicers = (west = (1, :, :),
           east = (grid.Nx, :, :),
           south = (:, 1, :),
           north = (:, grid.Ny, :),
           bottom = (:, :, 1),
           top = (:, :, grid.Nz))

for side in keys(slicers)
    indices = slicers[side]

    simulation.output_writers[side] = JLD2OutputWriter(model, (; b, u);
                                                       schedule = TimeInterval(save_fields_interval),
                                                       indices,
                                                       prefix = filename * "_$(side)_slice",
                                                       force = true)
end

simulation.output_writers[:zonal] = JLD2OutputWriter(model, (b=B, u=U);
                                                     schedule = TimeInterval(save_fields_interval),
                                                     prefix = filename * "_zonal_average",
                                                     force = true)

# Now let's run!

@info "Running the simulation..."

run!(simulation)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)

# ## Visualization

# Now we are ready to visualize our resutls! We use `CairoMakie` in this example.
# But on a system with OpenGL then `using GLMakie` can be more convenient.

using CairoMakie

filename = "baroclinic_adjustment"

fig = Figure(resolution = (1000, 800))
ax_b = fig[1, 1] = LScene(fig)

# Extract surfaces on all 6 boundaries

iter = Observable(0)

zonal_file = jldopen(filename * "_zonal_average.jld2")
grid = zonal_file["serialized/grid"]

sides = keys(slicers)

slice_files = NamedTuple(side => jldopen(filename * "_$(side)_slice.jld2") for side in sides)

# Build coordinates, rescaling the vertical coordinate

x, y, z = nodes((Center, Center, Center), grid)

yscale = 2.5
zscale = 600
z = z .* zscale
y = y .* yscale

zonal_slice_displacement = 1.5

# Plot buoyancy

b_slices = (
      west = @lift(Array(slice_files.west["timeseries/b/"   * string($iter)][1, :, :])),
      east = @lift(Array(slice_files.east["timeseries/b/"   * string($iter)][1, :, :])),
     south = @lift(Array(slice_files.south["timeseries/b/"  * string($iter)][:, 1, :])),
     north = @lift(Array(slice_files.north["timeseries/b/"  * string($iter)][:, 1, :])),
    bottom = @lift(Array(slice_files.bottom["timeseries/b/" * string($iter)][:, :, 1])),
       top = @lift(Array(slice_files.top["timeseries/b/"    * string($iter)][:, :, 1]))
)

clims_b = @lift 1.1 .* extrema(slice_files.top["timeseries/b/" * string($iter)][:])
kwargs_b = (colorrange=clims_b, colormap=:deep, show_axis=false)

surface!(ax_b, y, z, b_slices.west;   transformation = (:yz, x[1]),   kwargs_b...)
surface!(ax_b, y, z, b_slices.east;   transformation = (:yz, x[end]), kwargs_b...)
surface!(ax_b, x, z, b_slices.south;  transformation = (:xz, y[1]),   kwargs_b...)
surface!(ax_b, x, z, b_slices.north;  transformation = (:xz, y[end]), kwargs_b...)
surface!(ax_b, x, y, b_slices.bottom; transformation = (:xy, z[1]),   kwargs_b...)
surface!(ax_b, x, y, b_slices.top;    transformation = (:xy, z[end]), kwargs_b...)

b_avg = @lift zonal_file["timeseries/b/" * string($iter)][1, :, :]

surface!(ax_b, y, z, b_avg; transformation = (:yz, zonal_slice_displacement * x[end]), colorrange=clims_b, colormap=:deep)
contour!(ax_b, y, z, b_avg; levels = 15, linewidth=2, color=:black, transformation = (:yz, zonal_slice_displacement * x[end]), show_axis=false)

rotate_cam!(ax_b.scene, (π/20, -π/6, 0))

# Add a figure title with the time of the snapshot and then record a movie.

title = @lift(string("Buoyancy at t = ",
                     string(slice_files[1]["timeseries/t/" * string($iter)]/day), " days"))

fig[0, :] = Label(fig, title, textsize=50)

iterations = parse.(Int, keys(slice_files[1]["timeseries/t"]))

record(fig, filename * ".mp4", iterations, framerate=8) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    iter[] = i
end

# Let's now close all the files we opened.

for file in slice_files
    close(file)
end

close(zonal_file)
