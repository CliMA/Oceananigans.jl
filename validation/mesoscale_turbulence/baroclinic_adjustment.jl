using Printf
using Statistics
using GLMakie
using Random
using JLD2

using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: fields
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization

# Domain
Lx = 250kilometers # east-west extent [m]
Ly = 500kilometers # north-south extent [m]
Lz = 1kilometers    # depth [m]

Nx = 256
Ny = 512
Nz = 32

grid = RegularRectilinearGrid(topology = (Periodic, Bounded, Bounded), 
                              size = (Nx, Ny, Nz), 
                              x = (0, Lx),
                              y = (-Ly/2, Ly/2),
                              z = (-Lz, 0),
                              halo = (3, 3, 3))

coriolis = BetaPlane(latitude=-45)

Δx, Δy, Δz = Lx/Nx, Ly/Ny, Lz/Nz

𝒜 = Δz/Δx # Grid cell aspect ratio.

κh = 0.25   # [m²/s] horizontal diffusivity
νh = 0.25   # [m²/s] horizontal viscocity
κz = 𝒜 * κh # [m²/s] vertical diffusivity
νz = 𝒜 * νh # [m²/s] vertical viscocity

diffusive_closure = AnisotropicDiffusivity(νh = νh,
                                           νz = νz,
                                           κh = κh,
                                           κz = κz,
					                       time_discretization = VerticallyImplicitTimeDiscretization())

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
                                                                convective_νz = 0.0)

#####
##### Model building
#####

@info "Building a model..."

closures = (diffusive_closure, convective_adjustment)

model = HydrostaticFreeSurfaceModel(architecture = GPU(),
                                    grid = grid,
                                    coriolis = coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    closure = closures,
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
Δb = Δy * M²
ϵb = 1e-2 * Δb # noise amplitude

bᵢ(x, y, z) = N² * z + Δb * ramp(y, Δy) + ϵb * randn()
cᵢ(x, y, z) = exp(-y^2 / 2Δy^2)

set!(model, b=bᵢ, c=cᵢ)

#####
##### Simulation building
#####

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.8e, %6.8e, %6.8e) m/s, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            prettytime(sim.Δt.Δt))

    wall_clock[1] = time_ns()
    
    return nothing
end

wizard = TimeStepWizard(cfl=0.2, Δt=5minutes, max_Δt=5minutes)

simulation = Simulation(model, Δt=wizard, stop_time=40days, progress=print_progress, iteration_interval=100)

slicers = (west = FieldSlicer(i=1),
           east = FieldSlicer(i=grid.Nx),
           south = FieldSlicer(j=1),
           north = FieldSlicer(j=grid.Ny),
           bottom = FieldSlicer(k=1),
           top = FieldSlicer(k=grid.Nz))

for side in keys(slicers)
    field_slicer = slicers[side]

    simulation.output_writers[side] = JLD2OutputWriter(model, fields(model),
                                                       schedule = TimeInterval(1day),
                                                       field_slicer = field_slicer,
                                                       prefix = "baroclinic_adj_$(side)_slice",
                                                       force = true)
end

simulation.output_writers[:fields] = JLD2OutputWriter(model, fields(model),
                                                      schedule = TimeInterval(10day),
                                                      field_slicer = nothing,
                                                      prefix = "baroclinic_adj_fields",
                                                      force = true)

B = AveragedField(model.tracers.b, dims=1)
U = AveragedField(model.velocities.u, dims=1)
V = AveragedField(model.velocities.v, dims=1)
W = AveragedField(model.velocities.w, dims=1)

simulation.output_writers[:zonal] = JLD2OutputWriter(model, (b=B, u=U, v=V, w=W),
                                                     schedule = TimeInterval(1day),
                                                     prefix = "baroclinic_adj_zonal_average",
                                                     force = true)

@info "Running the simulation..."

run!(simulation, pickup=false)

@info "Simulation completed in " * prettytime(simulation.run_time)

fig = Figure(resolution = (1200, 800))
ax = fig[1, 1] = LScene(fig, title="Baroclinic Adjustment")

b_data = Array(interior(simulation.model.tracers.b))

# Extract surfaces on all 6 boundaries

iter = Node(0)
sides = keys(slicers)
slice_files = NamedTuple(side => jldopen("baroclinic_adj_$(side)_slice.jld2") for side in sides)
b_slices = NamedTuple(side => @lift(Array(slice_files[side]["timeseries/b/" * string($iter)])) for side in sides)

b_slices = (
      west = @lift(Array(slice_files.west["timeseries/b/"   * string($iter)][1, :, :])),
      east = @lift(Array(slice_files.east["timeseries/b/"   * string($iter)][1, :, :])),
     south = @lift(Array(slice_files.south["timeseries/b/"  * string($iter)][:, 1, :])),
     north = @lift(Array(slice_files.north["timeseries/b/"  * string($iter)][:, 1, :])),
    bottom = @lift(Array(slice_files.bottom["timeseries/b/" * string($iter)][:, :, 1])),
       top = @lift(Array(slice_files.top["timeseries/b/"    * string($iter)][:, :, 1]))
)


# b_west   = b_data[1,   :,   :]
# b_east   = b_data[end, :,   :]
# b_south  = b_data[:,   1,   :]
# b_north  = b_data[:, end,   :]
# b_bottom = b_data[:,   :,   1]
# b_top    = b_data[:,   :, end]

# Build coordinates, rescaling the vertical coordinate
x, y, z = nodes((Center, Center, Center), grid)

zscale = 100
z = z .* zscale

clims = @lift extrema(slice_files.top["timeseries/b/" * string($iter)][:])
kwargs = (colorrange=clims, colormap=:balance, show_axis=false)

GLMakie.surface!(ax, y, z, b_slices.west;   transformation = (:yz, x[1]),   kwargs...)
GLMakie.surface!(ax, y, z, b_slices.east;   transformation = (:yz, x[end]), kwargs...)
GLMakie.surface!(ax, x, z, b_slices.south;  transformation = (:xz, y[1]),   kwargs...)
GLMakie.surface!(ax, x, z, b_slices.north;  transformation = (:xz, y[end]), kwargs...)
GLMakie.surface!(ax, x, y, b_slices.bottom; transformation = (:xy, z[1]),   kwargs...)
GLMakie.surface!(ax, x, y, b_slices.top;    transformation = (:xy, z[end]), kwargs...)

zonal_file = jldopen("baroclinic_adj_zonal_average.jld2")

b_avg = @lift zonal_file["timeseries/b/" * string($iter)][1, :, :]
u_avg = @lift zonal_file["timeseries/u/" * string($iter)][1, :, :]

ulims = @lift extrema(zonal_file["timeseries/u/" * string($iter)][1, :, :])

GLMakie.contour!(ax, y, z, b_avg; levels = 10, transformation = (:yz, 1.5 * x[end]), show_axis=false)
GLMakie.surface!(ax, y, z, u_avg; transformation = (:yz, 1.5 * x[end]), colorrange=ulims, colormap=:balance)

title = @lift(string("Buoyancy and zonally-averaged u at t = ",
                     prettytime(zonal_file["timeseries/t/" * string($iter)])))

fig[0, 1] = Label(fig, title, textsize=30)

rotate_cam!(ax.scene, (π/24, -π/6, 0))

iterations = parse.(Int, keys(zonal_file["timeseries/t"]))

record(fig, "baroclinic_adjustment.mp4", iterations, framerate=12) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    iter[] = i
end

display(fig)

for file in slice_files
    close(file)
end

close(zonal_file)

