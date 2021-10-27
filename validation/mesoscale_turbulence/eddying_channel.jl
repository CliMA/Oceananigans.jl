#using Pkg
# pkg"add Oceananigans GLMakie JLD2"
# ENV["GKSwstype"] = "100"
# pushfirst!(LOAD_PATH, @__DIR__)
# pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "..")) # add Oceananigans

using Printf
using Statistics
# using GLMakie
using JLD2

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode

# Domain
const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]
const Lz = 2kilometers    # depth [m]

# number of grid points
Nx = 256
Ny = 512
Nz = 80

movie_interval = 2days
stop_time = 50years

arch = CPU()

# stretched grid

# we implement here a linearly streched grid in which the top grid cell has Δzₜₒₚ
# and every other cell is bigger by a factor σ, e.g.,
# Δzₜₒₚ, Δzₜₒₚ * σ, Δzₜₒₚ * σ², ..., Δzₜₒₚ * σᴺᶻ⁻¹,
# so that the sum of all cell heights is Lz

# Given Lz and stretching factor σ > 1 the top cell height is Δzₜₒₚ = Lz * (σ - 1) / σ^(Nz - 1)

σ = 1.04 # linear stretching factor
Δz_center_linear(k) = Lz * (σ - 1) * σ^(Nz - k) / (σ^Nz - 1) # k=1 is the bottom-most cell, k=Nz is the top cell
linearly_spaced_faces(k) = k==1 ? -Lz : - Lz + sum(Δz_center_linear.(1:k-1))

grid = VerticallyStretchedRectilinearGrid(architecture = arch,
                                          topology = (Periodic, Bounded, Bounded),
                                          size = (Nx, Ny, Nz),
                                          halo = (3, 3, 3),
                                          x = (0, Lx),
                                          y = (0, Ly),
                                          z_faces = linearly_spaced_faces)

# The vertical spacing versus depth for the prescribed grid
#=
plot(grid.Δzᵃᵃᶜ[1:Nz], grid.zᵃᵃᶜ[1:Nz],
     axis=(xlabel = "Vertical spacing (m)",
           ylabel = "Depth (m)"))
=#

@info "Built a grid: $grid."

#####
##### Boundary conditions
#####

α  = 2e-4     # [K⁻¹] thermal expansion coefficient 
g  = 9.8061   # [m s⁻²] gravitational constant
cᵖ = 3994.0   # [J K⁻¹] heat capacity
ρ  = 1024.0   # [kg m⁻³] reference density

parameters = (
              Ly = Ly,
              Lz = Lz,
              Qᵇ = 10/(ρ * cᵖ) * α * g,   # buoyancy flux magnitude [m² s⁻³]    
              y_shutoff = 5/6 * Ly,       # shutoff location for buoyancy flux [m]
              τ = 0.2 / ρ,                # surface kinematic wind stress [m² s⁻²]
              μ = 1 / 30days,             # bottom drag damping time-scale [s⁻¹]
              ΔB = 8 * α * g,             # surface vertical buoyancy gradient [s⁻²]
              H = Lz,                     # domain depth [m]
              h = 1000.0,                 # exponential decay scale of stable stratification [m]
              y_sponge = 19/20 * Ly,      # southern boundary of sponge layer [m]
              λt = 7days                  # relaxation time scale [s]
              )

@inline function buoyancy_flux(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return ifelse(y < p.y_shutoff, p.Qᵇ * cos(3π * y / p.Ly), 0.0)
end

buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, discrete_form=true, parameters=parameters)


@inline function u_stress(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return - p.τ * sin(π * y / p.Ly)
end

u_stress_bc = FluxBoundaryCondition(u_stress, discrete_form=true, parameters=parameters)

@inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds - p.μ * p.Lz * model_fields.u[i, j, 1] 
@inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds - p.μ * p.Lz * model_fields.v[i, j, 1]

u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form=true, parameters=parameters)
v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form=true, parameters=parameters)

b_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc)

u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
v_bcs = FieldBoundaryConditions(bottom = v_drag_bc)

#####
##### Coriolis
#####

const f = -1e-4     # [s⁻¹]
const β =  1e-11    # [m⁻¹ s⁻¹]
coriolis = BetaPlane(f₀ = f, β = β)

#####
##### Forcing and initial condition
#####

@inline initial_buoyancy(z, p) = p.ΔB * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))
@inline mask(y, p) = max(0.0, y - p.y_sponge) / (Ly - p.y_sponge)

@inline function buoyancy_relaxation(i, j, k, grid, clock, model_fields, p)
    timescale = p.λt
    y = ynode(Center(), j, grid)
    z = znode(Center(), k, grid)
    target_b = initial_buoyancy(z, p)
    b = @inbounds model_fields.b[i, j, k]
    return - 1 / timescale  * mask(y, p) * (b - target_b)
end

Fb = Forcing(buoyancy_relaxation, discrete_form = true, parameters = parameters)

# Turbulence closures

κh = 0.5e-5 # [m²/s] horizontal diffusivity
νh = 30.0   # [m²/s] horizontal viscocity
κz = 0.5e-5 # [m²/s] vertical diffusivity
νz = 3e-4   # [m²/s] vertical viscocity

horizontal_diffusivity = AnisotropicDiffusivity(νh=νh, νz=νz, κh=κh, κz=κz)

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
                                                                convective_νz = 0.0)

#####
##### Model building
#####

@info "Building a model..."

model = HydrostaticFreeSurfaceModel(architecture = arch,
                                    grid = grid,
                                    free_surface = ImplicitFreeSurface(),
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = coriolis,
                                    closure = (horizontal_diffusivity, convective_adjustment),
                                    tracers = :b,
                                    boundary_conditions = (b=b_bcs, u=u_bcs, v=v_bcs),
                                    forcing = (; b=Fb,)
                                    )


@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
ε(σ) = σ * randn()
bᵢ(x, y, z) = parameters.ΔB * ( exp(z/parameters.h) - exp(-Lz/parameters.h) ) / (1 - exp(-Lz/parameters.h)) + ε(1e-8)
uᵢ(x, y, z) = ε(1e-8)
vᵢ(x, y, z) = ε(1e-8)
wᵢ(x, y, z) = ε(1e-8)

set!(model, b=bᵢ, u=uᵢ, v=vᵢ, w=wᵢ)

#####
##### Simulation building

wizard = TimeStepWizard(cfl=0.1, Δt=5minutes, max_change=1.1, max_Δt=20minutes)

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
            prettytime(sim.Δt.Δt))
 #           prettytime(sim.Δt))

    wall_clock[1] = time_ns()
    
    return nothing
end

simulation = Simulation(model, Δt=wizard, stop_time=stop_time, progress=print_progress, iteration_interval=10)

#####
##### Diagnostics
#####

u, v, w = model.velocities
b = model.tracers.b

ζ = ComputedField(∂x(v) - ∂y(u))

B = AveragedField(b, dims=1)
U = AveragedField(u, dims=1)
V = AveragedField(v, dims=1)
W = AveragedField(w, dims=1)

b′ = b - B
v′ = v - V
w′ = w - W

v′b′ = AveragedField(v′ * b′, dims=1)
w′b′ = AveragedField(w′ * b′, dims=1)

outputs = (; b, ζ, u)

averaged_outputs = (; v′b′, w′b′, B, U)

#####
##### Build checkpointer and output writer
#####

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(1years),
                                                        prefix = "eddying_channel",
                                                        force = true)

slicers = (west = FieldSlicer(i=1),
           east = FieldSlicer(i=grid.Nx),
           south = FieldSlicer(j=1),
           north = FieldSlicer(j=grid.Ny),
           bottom = FieldSlicer(k=1),
           top = FieldSlicer(k=grid.Nz))

for side in keys(slicers)
    field_slicer = slicers[side]

    simulation.output_writers[side] = JLD2OutputWriter(model, outputs,
                                                       schedule = TimeInterval(movie_interval),
                                                       field_slicer = field_slicer,
                                                       prefix = "eddying_channel_$(side)_slice",
                                                       force = true)
end

simulation.output_writers[:zonal] = JLD2OutputWriter(model, (b=B, u=U),#, v=V, w=W, vb=v′b′, wb=w′b′),
                                                     schedule = TimeInterval(movie_interval),
                                                     prefix = "eddying_channel_zonal_average",
                                                     force = true)
#=
simulation.output_writers[:averages] = JLD2OutputWriter(model, averaged_outputs,
                                                        schedule = AveragedTimeInterval(1days, window=1days, stride=1),
                                                        prefix = "eddying_channel_averages",
                                                        verbose = true,
                                                        force = true)
=#

 @info "Running the simulation..."

try
    run!(simulation, pickup=false)
catch err
    @info "run! threw an error! The error message is"
    showerror(stdout, err)
end


using GLMakie

#####
##### Visualize
#####

fig = Figure(resolution = (2000, 1000))
ax_b = fig[1:5, 1] = LScene(fig)
ax_ζ = fig[1:5, 2] = LScene(fig)

# Extract surfaces on all 6 boundaries

iter = Node(0)
sides = keys(slicers)

zonal_file = jldopen("eddying_channel_zonal_average.jld2")
slice_files = NamedTuple(side => jldopen("eddying_channel_$(side)_slice.jld2") for side in sides)

grid = VerticallyStretchedRectilinearGrid(architecture = CPU(),
                                          topology = (Periodic, Bounded, Bounded),
                                          size = (Nx, Ny, Nz),
                                          halo = (3, 3, 3),
                                          x = (0, Lx),
                                          y = (0, Ly),
                                          z_faces = linearly_spaced_faces)

# Build coordinates, rescaling the vertical coordinate

xζ, yζ, zζ = nodes((Face, Face, Center), grid)
xu, yu, zu = nodes((Face, Center, Center), grid)
xv, yv, zv = nodes((Center, Face, Center), grid)
xw, yw, zw = nodes((Center, Center, Face), grid)
xb, yb, zb = nodes((Center, Center, Center), grid)

zscale = 300
zu = zu .* zscale
zb = zb .* zscale
zζ = zζ .* zscale

zonal_slice_displacement = 1.35

b_slices = (
      west = @lift(Array(slice_files.west["timeseries/b/"   * string($iter)][1, :, :])),
      east = @lift(Array(slice_files.east["timeseries/b/"   * string($iter)][1, :, :])),
     south = @lift(Array(slice_files.south["timeseries/b/"  * string($iter)][:, 1, :])),
     north = @lift(Array(slice_files.north["timeseries/b/"  * string($iter)][:, 1, :])),
    bottom = @lift(Array(slice_files.bottom["timeseries/b/" * string($iter)][:, :, 1])),
       top = @lift(Array(slice_files.top["timeseries/b/"    * string($iter)][:, :, 1]))
)

clims_b = @lift extrema(slice_files.top["timeseries/b/" * string($iter)][:])
kwargs_b = (colorrange=clims_b, colormap=:deep, show_axis=false)

GLMakie.surface!(ax_b, yb, zb, b_slices.west;   transformation = (:yz, xb[1]),   kwargs_b...)
GLMakie.surface!(ax_b, yb, zb, b_slices.east;   transformation = (:yz, xb[end]), kwargs_b...)
GLMakie.surface!(ax_b, xb, zb, b_slices.south;  transformation = (:xz, yb[1]),   kwargs_b...)
GLMakie.surface!(ax_b, xb, zb, b_slices.north;  transformation = (:xz, yb[end]), kwargs_b...)
GLMakie.surface!(ax_b, xb, yb, b_slices.bottom; transformation = (:xy, zb[1]),   kwargs_b...)
GLMakie.surface!(ax_b, xb, yb, b_slices.top;    transformation = (:xy, zb[end]), kwargs_b...)

b_avg = @lift zonal_file["timeseries/b/" * string($iter)][1, :, :]
u_avg = @lift zonal_file["timeseries/u/" * string($iter)][1, :, :]

clims_u = @lift extrema(zonal_file["timeseries/u/" * string($iter)][1, :, :])
clims_u = (-0.4, 0.4)

GLMakie.contour!(ax_b, yb, zb, b_avg; levels = 25, color = :black, linewidth = 2, transformation = (:yz, zonal_slice_displacement * xb[end]), show_axis=false)
GLMakie.surface!(ax_b, yu, zu, u_avg; transformation = (:yz, zonal_slice_displacement * xu[end]), colorrange=clims_u, colormap=:balance)

rotate_cam!(ax_b.scene, (π/24, -π/6, 0))

ζ_slices = (
      west = @lift(Array(slice_files.west["timeseries/ζ/"   * string($iter)][1, :, :])),
      east = @lift(Array(slice_files.east["timeseries/ζ/"   * string($iter)][1, :, :])),
     south = @lift(Array(slice_files.south["timeseries/ζ/"  * string($iter)][:, 1, :])),
     north = @lift(Array(slice_files.north["timeseries/ζ/"  * string($iter)][:, 1, :])),
    bottom = @lift(Array(slice_files.bottom["timeseries/ζ/" * string($iter)][:, :, 1])),
       top = @lift(Array(slice_files.top["timeseries/ζ/"    * string($iter)][:, :, 1]))
)

u_slices = (
      west = @lift(Array(slice_files.west["timeseries/u/"   * string($iter)][1, :, :])),
      east = @lift(Array(slice_files.east["timeseries/u/"   * string($iter)][1, :, :])),
     south = @lift(Array(slice_files.south["timeseries/u/"  * string($iter)][:, 1, :])),
     north = @lift(Array(slice_files.north["timeseries/u/"  * string($iter)][:, 1, :])),
    bottom = @lift(Array(slice_files.bottom["timeseries/u/" * string($iter)][:, :, 1])),
       top = @lift(Array(slice_files.top["timeseries/u/"    * string($iter)][:, :, 1]))
)

clims_ζ = @lift extrema(slice_files.top["timeseries/ζ/" * string($iter)][:])
clims_ζ = (-1f-4, 1f-4)
kwargs_ζ = (colormap=:curl, show_axis=false, colorrange=clims_ζ)
kwargs_east = (colormap=:curl, show_axis=false, colorrange=(-5f-5, 5f-5))
clims_u = @lift extrema(slice_files.top["timeseries/u/" * string($iter)][:])
kwargs_u = (colormap=:balance, show_axis=false)

GLMakie.surface!(ax_ζ, yζ, zζ, ζ_slices.west;   transformation = (:yz, xζ[1]),   kwargs_ζ...)
GLMakie.surface!(ax_ζ, yζ, zζ, ζ_slices.east;   transformation = (:yz, xζ[end]), kwargs_east...)
GLMakie.surface!(ax_ζ, xζ, zζ, ζ_slices.south;  transformation = (:xz, yζ[1]),   kwargs_ζ...)
GLMakie.surface!(ax_ζ, xζ, zζ, ζ_slices.north;  transformation = (:xz, yζ[end]), kwargs_ζ...)
GLMakie.surface!(ax_ζ, xζ, yζ, ζ_slices.bottom; transformation = (:xy, zζ[1]),   kwargs_ζ...)
GLMakie.surface!(ax_ζ, xζ, yζ, ζ_slices.top;    transformation = (:xy, zζ[end]), kwargs_ζ...)

b_avg = @lift zonal_file["timeseries/b/" * string($iter)][1, :, :]
u_avg = @lift zonal_file["timeseries/u/" * string($iter)][1, :, :]

clims_u = @lift extrema(zonal_file["timeseries/u/" * string($iter)][1, :, :])
clims_u = (-0.4, 0.4)

GLMakie.contour!(ax_ζ, yb, zb, b_avg; levels = 25, color = :black, linewidth = 2, transformation = (:yz, zonal_slice_displacement * xb[end]), show_axis=false)
GLMakie.surface!(ax_ζ, yu, zu, u_avg; transformation = (:yz, zonal_slice_displacement * xu[end]), colorrange=clims_u, colormap=:balance)

rotate_cam!(ax_ζ.scene, (π/24, -π/6, 0))

title = @lift(string("Buoyancy, vertical vorticity, and zonally-averaged u at t = ",
                     prettytime(zonal_file["timeseries/t/" * string($iter)])))

fig[0, :] = Label(fig, title, textsize=30)

display(fig)

iterations = parse.(Int, keys(zonal_file["timeseries/t"]))
iterations = iterations[1:Int((length(iterations)+1)/4)]

record(fig, "eddying_channel.mp4", iterations, framerate=12) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    iter[] = i
end

display(fig)

for file in slice_files
    close(file)
end

close(zonal_file)
