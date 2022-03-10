using Printf
using Statistics
using Random
using JLD2

using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: fields

arch = CPU()

filename = "baroclinic_adjustment"

# Domain
Lx = 1000kilometers # east-west extent [m]
Ly = 1000kilometers # north-south extent [m]
Lz = 1kilometers    # depth [m]

Nx = 64
Ny = 64
Nz = 40

save_fields_interval = 0.5day
stop_time = 40days
Δt₀ = 5minutes

grid = RectilinearGrid(arch;
                       topology = (Periodic, Bounded, Bounded), 
                       size = (Nx, Ny, Nz), 
                       x = (0, Lx),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0),
                       halo = (3, 3, 3))

coriolis = BetaPlane(latitude = -45)

Δx, Δy, Δz = Lx/Nx, Ly/Ny, Lz/Nz

𝒜 = Δz/Δx # Grid cell aspect ratio.

κh = 0.1    # [m² s⁻¹] horizontal diffusivity
νh = 0.1    # [m² s⁻¹] horizontal viscosity
κz = 𝒜 * κh # [m² s⁻¹] vertical diffusivity
νz = 𝒜 * νh # [m² s⁻¹] vertical viscosity

vertical_diffusive_closure = VerticalScalarDiffusivity(ν = νz, κ = κz)

horizontal_diffusive_closure = HorizontalScalarDiffusivity(ν = νh, κ = κh)

#####
##### Model building
#####

@info "Building a model..."

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    coriolis = coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    closure = (vertical_diffusive_closure, horizontal_diffusive_closure),
                                    tracers = (:b, :c),
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    free_surface = ImplicitFreeSurface())

@info "Built $model."

#####
##### Initial conditions
#####

"""
Linear ramp from 0 to 1 between -Δy/2 and +Δy/2.

For example:

y < y₀           => ramp = 0
y₀ < y < y₀ + Δy => ramp = y / Δy
y > y₀ + Δy      => ramp = 1
"""
ramp(y, Δy) = min(max(0, y/Δy + 1/2), 1)

# Parameters
N² = 4e-6 # [s⁻²] buoyancy frequency / stratification
M² = 8e-8 # [s⁻²] horizontal buoyancy gradient

Δy = 50kilometers
Δz = 100

Δc = 2Δy
Δb = Δy * M²
ϵb = 1e-2 * Δb # noise amplitude

bᵢ(x, y, z) = N² * z + Δb * ramp(y, Δy) + ϵb * randn()
cᵢ(x, y, z) = exp(-y^2 / 2Δc^2) * exp(-(z + Lz/4)^2 / 2Δz^2)

set!(model, b=bᵢ, c=cᵢ)

#####
##### Simulation building
#####

simulation = Simulation(model, Δt=Δt₀, stop_time=stop_time)

# add timestep wizard callback
wizard = TimeStepWizard(cfl=0.2, max_change=1.1, max_Δt=20minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

# add progress callback
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


#####
##### Diagnostics
#####

u, v, w = model.velocities
b, c = model.tracers.b, model.tracers.c

ζ = Field(∂x(v) - ∂y(u))

B = Field(Average(b, dims=1))
C = Field(Average(c, dims=1))
U = Field(Average(u, dims=1))
V = Field(Average(v, dims=1))
W = Field(Average(w, dims=1))

b′ = b - B
v′ = v - V
w′ = w - W

v′b′ = Field(Average(v′ * b′, dims=1))
w′b′ = Field(Average(w′ * b′, dims=1))

outputs = (; b, c, ζ, u, v, w)

averaged_outputs = (; v′b′, w′b′, B, U)


#####
##### Build checkpointer and output writer
#####

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(5years),
                                                        prefix = filename,
                                                        force = true)

slicers = (west = (1, :, :),
           east = (grid.Nx, :, :),
           south = (:, 1, :),
           north = (:, grid.Ny, :),
           bottom = (:, :, 1),
           top = (:, :, grid.Nz))

for side in keys(slicers)
    indices = slicers[side]

    simulation.output_writers[side] = JLD2OutputWriter(model, outputs;
                                                       schedule = TimeInterval(save_fields_interval),
                                                       indices,
                                                       prefix = filename * "_$(side)_slice",
                                                       force = true)
end

simulation.output_writers[:zonal] = JLD2OutputWriter(model, (b=B, c=C, u=U, v=V, w=W, vb=v′b′, wb=w′b′),
                                                     schedule = TimeInterval(save_fields_interval),
                                                     prefix = filename * "_zonal_average",
                                                     force = true)

@info "Running the simulation..."

run!(simulation)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)


#####
##### Visualize
#####

using CairoMakie

using Oceananigans, JLD2

filename = "baroclinic_adjustment"

fig = Figure(resolution = (3000, 1600))

ax_b = fig[1:5, 1] = LScene(fig)
ax_c = fig[1:5, 2] = LScene(fig)

axis_rotation_angles = (π/24, -π/6, 0)


# Extract surfaces on all 6 boundaries

iter = Observable(0)

zonal_file = jldopen(filename * "_zonal_average.jld2")
grid = zonal_file["serialized/grid"]

sides = keys(slicers)

slice_files = NamedTuple(side => jldopen(filename * "_$(side)_slice.jld2") for side in sides)

# Build coordinates, rescaling the vertical coordinate
x, y, z = nodes((Center, Center, Center), grid)

yscale = 3
zscale = 800
z = z .* zscale
y = y .* yscale

zonal_slice_displacement = 1.35

#####
##### Plot buoyancy...
#####

b_slices = (
      west = @lift(Array(slice_files.west["timeseries/b/"   * string($iter)][1, :, :])),
      east = @lift(Array(slice_files.east["timeseries/b/"   * string($iter)][1, :, :])),
     south = @lift(Array(slice_files.south["timeseries/b/"  * string($iter)][:, 1, :])),
     north = @lift(Array(slice_files.north["timeseries/b/"  * string($iter)][:, 1, :])),
    bottom = @lift(Array(slice_files.bottom["timeseries/b/" * string($iter)][:, :, 1])),
       top = @lift(Array(slice_files.top["timeseries/b/"    * string($iter)][:, :, 1]))
)

clims_b = @lift 1.1 .* extrema(slice_files.top["timeseries/b/" * string($iter)][:])
kwargs_b = (colorrange=clims_b, colormap=:balance, show_axis=false)

surface!(ax_b, y, z, b_slices.west;   transformation = (:yz, x[1]),   kwargs_b...)
surface!(ax_b, y, z, b_slices.east;   transformation = (:yz, x[end]), kwargs_b...)
surface!(ax_b, x, z, b_slices.south;  transformation = (:xz, y[1]),   kwargs_b...)
surface!(ax_b, x, z, b_slices.north;  transformation = (:xz, y[end]), kwargs_b...)
surface!(ax_b, x, y, b_slices.bottom; transformation = (:xy, z[1]),   kwargs_b...)
surface!(ax_b, x, y, b_slices.top;    transformation = (:xy, z[end]), kwargs_b...)

b_avg = @lift zonal_file["timeseries/b/" * string($iter)][1, :, :]
u_avg = @lift zonal_file["timeseries/u/" * string($iter)][1, :, :]

clims_u = @lift extrema(zonal_file["timeseries/u/" * string($iter)][1, :, :])

surface!(ax_b, y, z, b_avg; transformation = (:yz, zonal_slice_displacement * x[end]), colorrange=clims_b, colormap=:balance)
contour!(ax_b, y, z, b_avg; levels = 15, linewidth=2, color=:black, transformation = (:yz, zonal_slice_displacement * x[end]), show_axis=false)

rotate_cam!(ax_b.scene, axis_rotation_angles)

#####
##### Plot tracer...
#####

c_slices = (
      west = @lift(Array(slice_files.west["timeseries/c/"   * string($iter)][1, :, :])),
      east = @lift(Array(slice_files.east["timeseries/c/"   * string($iter)][1, :, :])),
     south = @lift(Array(slice_files.south["timeseries/c/"  * string($iter)][:, 1, :])),
     north = @lift(Array(slice_files.north["timeseries/c/"  * string($iter)][:, 1, :])),
    bottom = @lift(Array(slice_files.bottom["timeseries/c/" * string($iter)][:, :, 1])),
       top = @lift(Array(slice_files.top["timeseries/c/"    * string($iter)][:, :, 1]))
)

clims_c = @lift extrema(slice_files.west["timeseries/c/" * string($iter)][:])
clims_c = (0, 0.5)
kwargs_c = (colorrange=clims_c, colormap=:deep, show_axis=false)

surface!(ax_c, y, z, c_slices.west;   transformation = (:yz, x[1]),   kwargs_c...)
surface!(ax_c, y, z, c_slices.east;   transformation = (:yz, x[end]), kwargs_c...)
surface!(ax_c, x, z, c_slices.south;  transformation = (:xz, y[1]),   kwargs_c...)
surface!(ax_c, x, z, c_slices.north;  transformation = (:xz, y[end]), kwargs_c...)
surface!(ax_c, x, y, c_slices.bottom; transformation = (:xy, z[1]),   kwargs_c...)
surface!(ax_c, x, y, c_slices.top;    transformation = (:xy, z[end]), kwargs_c...)

b_avg = @lift zonal_file["timeseries/b/" * string($iter)][1, :, :]
c_avg = @lift zonal_file["timeseries/c/" * string($iter)][1, :, :]

surface!(ax_c, y, z, c_avg; transformation = (:yz, zonal_slice_displacement * x[end]), colorrange=clims_c, colormap=:deep)
contour!(ax_c, y, z, b_avg; levels = 15, linewidth=2, color=:black, transformation = (:yz, zonal_slice_displacement * x[end]), show_axis=false)

rotate_cam!(ax_c.scene, axis_rotation_angles)

#####
##### Make title and animate
#####

title = @lift(string("Buoyancy and tracer concentration at t = ",
                     string(slice_files[1]["timeseries/t/" * string($iter)]/day), " days"))

fig[0, :] = Label(fig, title, textsize=60)


iterations = parse.(Int, keys(slice_files[1]["timeseries/t"]))

# display(fig)

record(fig, filename * ".mp4", iterations, framerate=8) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    iter[] = i
end

for file in slice_files
    close(file)
end

close(zonal_file)
