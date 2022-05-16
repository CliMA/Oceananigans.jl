ENV["GKSwstype"] = "100"

using Printf
using Statistics
using Random
using JLD2

using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels: Fields

filename = "baroclinic_adjustment"

# Architecture
architecture  = CPU()

# Domain
Lx = 1000kilometers  # east-west extent [m]
Ly = 1000kilometers  # north-south extent [m]
Lz = 1kilometers     # depth [m]

Nx = 128
Ny = 128
Nz = 16

save_fields_interval = 1day
stop_time = 60days
Δt₀ = 5minutes

# We choose a regular grid though because of numerical issues that yet need to be resolved
grid = RectilinearGrid(architecture,
                       topology = (Periodic, Bounded, Bounded), 
                       size = (Nx, Ny, Nz), 
                       x = (0, Lx),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0),
                       halo = (3, 3, 3))

coriolis = BetaPlane(latitude = -45)

#####
##### Model building
#####

@info "Building a model..."

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    coriolis = coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    closure = nothing,
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

Nx, Ny, Nz = size(grid)

slice_indices = (west   = (1,   :,  :),
                 east   = (Nx,  :,  :),
                 south  = (:,   1,  :),
                 north  = (:,  Ny,  :),
                 bottom = (:,   :,  1),
                 top    = (:,   :, Nz))

#=
for side in keys(slice_indices)
    indices = slice_indices[side]

    simulation.output_writers[side] = JLD2OutputWriter(model, fields(model); indices,
                                                       schedule = TimeInterval(save_fields_interval),
                                                       filename = filename * "_$(side)_slice",
                                                       overwrite_existing = true)
end

simulation.output_writers[:fields] = JLD2OutputWriter(model, fields(model),
                                                      schedule = TimeInterval(save_fields_interval),
                                                      filename = filename * "_fields",
                                                      overwrite_existing = true)

B = Average(model.tracers.b, dims=1)
C = Average(model.tracers.c, dims=1)
U = Average(model.velocities.u, dims=1)
V = Average(model.velocities.v, dims=1)
W = Average(model.velocities.w, dims=1)

simulation.output_writers[:zonal] = JLD2OutputWriter(model, (b=B, c=C, u=U, v=V, w=W),
                                                     schedule = TimeInterval(save_fields_interval),
                                                     filename = filename * "_zonal_average",
                                                     overwrite_existing = true)

@info "Running the simulation..."

run!(simulation, pickup=false)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)
=#

#####
##### Visualize
#####

using CairoMakie

fig = Figure(resolution = (1400, 700))
ax_b = fig[1:5, 1] = LScene(fig)
ax_c = fig[1:5, 2] = LScene(fig)

# Extract surfaces on all 6 boundaries
iter = Node(0)
sides = keys(slice_indices)

zonal_file = jldopen(filename * "_zonal_average.jld2")
slice_files = NamedTuple(side => jldopen(filename * "_$(side)_slice.jld2") for side in sides)

grid = slice_files[1]["serialized/grid"]

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

clims_b = @lift extrema(slice_files.top["timeseries/b/" * string($iter)][:])
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

contour!(ax_b, y, z, b_avg; levels = 25, linewidth=2, color=:black, transformation = (:yz, zonal_slice_displacement * x[end]), show_axis=false)
surface!(ax_b, y, z, u_avg; transformation = (:yz, zonal_slice_displacement * x[end]), colorrange=clims_u, colormap=:balance)

rotate_cam!(ax_b.scene, (π/24, -π/6, 0))

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

clims_c = @lift extrema(slice_files.top["timeseries/c/" * string($iter)][:])
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

contour!(ax_c, y, z, b_avg; levels = 25, linewidth=2, color=:black, transformation = (:yz, zonal_slice_displacement * x[end]), show_axis=false)
surface!(ax_c, y, z, c_avg; transformation = (:yz, zonal_slice_displacement * x[end]), colorrange=clims_c, colormap=:deep)

rotate_cam!(ax_c.scene, (π/24, -π/6, 0))

#####
##### Make title and animate
#####

title = @lift(string("Buoyancy and tracer concentration at t = ",
                     prettytime(slice_files[1]["timeseries/t/" * string($iter)])))

fig[0, :] = Label(fig, title, textsize=30)

display(fig)

iterations = parse.(Int, keys(slice_files[1]["timeseries/t"]))

record(fig, filename * ".mp4", iterations, framerate=8) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    iter[] = i
end

display(fig)

for file in slice_files
    close(file)
end

close(zonal_file)
