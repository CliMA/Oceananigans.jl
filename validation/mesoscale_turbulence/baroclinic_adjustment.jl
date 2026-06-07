ENV["GKSwstype"] = "100"

using Printf
using Statistics
using Random
using JLD2

using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions
using Oceananigans.TurbulenceClosures

Oceananigans.defaults.FloatType = Float32
filename = "baroclinic_adjustment"

# Architecture
# arch = GPU(metal)
arch = CPU()

# Domain
const Lx = 2000kilometers  # east-west extent [m]
const Ly = 1000kilometers  # north-south extent [m]
const Lz = 1kilometers     # depth [m]

const Nx = 256
const Ny = 128
const Nz = 32

save_fields_interval = 1day
stop_time = 80days
Δt = 10minutes

# We choose a regular grid though because of numerical issues that yet need to be resolved
grid = RectilinearGrid(arch,
                       topology = (Bounded, Bounded, Bounded),
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0),
                       halo = (3, 3, 3))

coriolis = BetaPlane(latitude = -45)

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
const N² = 4e-6 # [s⁻²] buoyancy frequency / stratification
const M² = 8e-8 # [s⁻²] horizontal buoyancy gradient

const Δy = 50kilometers
const Δz = 100

const Δc = 2Δy
const Δb = Δy * M²
const ϵb = 1e-2 * Δb # noise amplitude

@inline bᵢ(x, y, z) = N² * z + Δb * ramp(y, Δy) + ϵb * randn()
@inline cᵢ(x, y, z) = exp(-y^2 / 2Δc^2) * exp(-(z + Lz/4)^2 / 2Δz^2)

@info "Building a model..."

@inline bᵂ(y, z, t) = bᵢ(0,  y, z)
@inline bᴱ(y, z, t) = bᵢ(Lx, y, z)

@inline cᵂ(y, z, t) = cᵢ(0,  y, z)
@inline cᴱ(y, z, t) = cᵢ(Lx, y, z)

Ubcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing), 
                               west = FlatherBoundaryCondition(0, 0),
                               east = FlatherBoundaryCondition(0, 0))

ubcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0; scheme=Radiation()),
                               east = NormalFlowBoundaryCondition(0; scheme=Radiation()))

vbcs = FieldBoundaryConditions(west = ValueBoundaryCondition(0; scheme=Radiation()),
                               east = ValueBoundaryCondition(0; scheme=Radiation()))

bbcs = FieldBoundaryConditions(west = ValueBoundaryCondition(bᵂ; scheme=Radiation()),
                               east = ValueBoundaryCondition(bᴱ; scheme=Radiation()))

cbcs = FieldBoundaryConditions(west = ValueBoundaryCondition(cᵂ; scheme=Radiation()),
                               east = ValueBoundaryCondition(cᴱ; scheme=Radiation()))

model = HydrostaticFreeSurfaceModel(grid;
                                    coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    tracers = (:b, :c),
                                    momentum_advection = WENO(order=5),
                                    tracer_advection = WENO(order=5),
                                    boundary_conditions = (U = Ubcs, u=ubcs, v=vbcs, b=bbcs, c=cbcs),
                                    free_surface = SplitExplicitFreeSurface(grid; substeps=60))

@info "Built $model."

set!(model, b=bᵢ, c=cᵢ)

#####
##### Simulation building
#####

simulation = Simulation(model; Δt, stop_time)

# add timestep wizard callback
# wizard = TimeStepWizard(cfl=0.2, max_change=1.1, max_Δt=20minutes)
# simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

# add progress callback
wall_clock = Ref(time_ns())

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            prettytime(sim.Δt))

    wall_clock[] = time_ns()

    return nothing
end

add_callback!(simulation, print_progress, IterationInterval(100))


slicers = (west = (1, :, :),
           east = (grid.Nx, :, :),
           south = (:, 1, :),
           north = (:, grid.Ny, :),
           bottom = (:, :, 1),
           top = (:, :, grid.Nz))

for side in keys(slicers)
    indices = slicers[side]
    simulation.output_writers[side] = JLD2Writer(model, merge(model.velocities, model.tracers);
                                                 schedule = TimeInterval(save_fields_interval),
                                                 indices,
                                                 filename = filename * "_$(side)_slice",
                                                 overwrite_existing = true)
end

simulation.output_writers[:fields] = JLD2Writer(model, fields(model),
                                                schedule = TimeInterval(save_fields_interval),
                                                filename = filename * "_fields",
                                                overwrite_existing = true)

B = Field(Average(model.tracers.b, dims=1))
C = Field(Average(model.tracers.c, dims=1))
U = Field(Average(model.velocities.u, dims=1))
V = Field(Average(model.velocities.v, dims=1))
W = Field(Average(model.velocities.w, dims=1))

simulation.output_writers[:zonal] = JLD2Writer(model, (b=B, c=C, u=U, v=V, w=W),
                                               schedule = TimeInterval(save_fields_interval),
                                               filename = filename * "_zonal_average",
                                               overwrite_existing = true)


@info "Running the simulation..."

run!(simulation)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)

#####
##### Visualize
#####

using CairoMakie

fig = Figure(size=(1400, 900))
ax_b = Axis(fig[1, 1])
ax_c = Axis(fig[2, 1])
ax_u = Axis(fig[3, 1])

# Extract surfaces on all 6 boundaries
iter = Observable(0)
sides = keys(slicers)

zonal_file = jldopen(filename * "_zonal_average.jld2")
slice_files = NamedTuple(side => jldopen(filename * "_$(side)_slice.jld2") for side in sides)

grid = slice_files[1]["serialized/grid"]

# Build coordinates, rescaling the vertical coordinate
x, y, z = nodes(grid, Center(), Center(), Center())

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

c_slices = (
      west = @lift(Array(slice_files.west["timeseries/c/"   * string($iter)][1, :, :])),
      east = @lift(Array(slice_files.east["timeseries/c/"   * string($iter)][1, :, :])),
     south = @lift(Array(slice_files.south["timeseries/c/"  * string($iter)][:, 1, :])),
     north = @lift(Array(slice_files.north["timeseries/c/"  * string($iter)][:, 1, :])),
    bottom = @lift(Array(slice_files.bottom["timeseries/c/" * string($iter)][:, :, 1])),
       top = @lift(Array(slice_files.top["timeseries/c/"    * string($iter)][:, :, 1]))
)

u_slices = (
      west = @lift(Array(slice_files.west["timeseries/u/"   * string($iter)][1, :, :])),
      east = @lift(Array(slice_files.east["timeseries/u/"   * string($iter)][1, :, :])),
     south = @lift(Array(slice_files.south["timeseries/u/"  * string($iter)][:, 1, :])),
     north = @lift(Array(slice_files.north["timeseries/u/"  * string($iter)][:, 1, :])),
    bottom = @lift(Array(slice_files.bottom["timeseries/u/" * string($iter)][:, :, 1])),
       top = @lift(Array(slice_files.top["timeseries/u/"    * string($iter)][:, :, 1]))
)

clims_b = @lift extrema(slice_files.top["timeseries/b/" * string($iter)][:])
kwargs_b = (colorrange=clims_b, colormap=:balance, show_axis=false)

heatmap!(ax_b, x, y, b_slices.top)
heatmap!(ax_c, x, y, c_slices.top)
heatmap!(ax_u, x, y, u_slices.top)

#####
##### Make title and animate
#####

title = @lift(string("Buoyancy and tracer concentration at t = ",
                     prettytime(slice_files[1]["timeseries/t/" * string($iter)])))

fig[0, :] = Label(fig, title, fontsize=30)


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
