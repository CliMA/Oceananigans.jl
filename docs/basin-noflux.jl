#using Pkg
# pkg"add Oceananigans GLMakie JLD2"
# ENV["GKSwstype"] = "100"
# pushfirst!(LOAD_PATH, @__DIR__)
# pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "..")) # add Oceananigans

using Printf
using Statistics
# using GLMakie
using JLD2
using CUDA

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary

# Domain
const Lx = 10kilometers # zonal domain length [m]
const Ly = 20kilometers # meridional domain length [m]
const Lz = 1kilometers    # depth [m]

# number of grid points
Nx = 256
Ny = 512
Nz = 128 

movie_interval    = 1minutes
stop_time         = 2days 
checkpointer_time = 12hours

arch = GPU()

# stretched grid

# we implement here a linearly streched grid in which the top grid cell has Δzₜₒₚ
# and every other cell is bigger by a factor σ, e.g.,
# Δzₜₒₚ, Δzₜₒₚ * σ, Δzₜₒₚ * σ², ..., Δzₜₒₚ * σᴺᶻ⁻¹,
# so that the sum of all cell heights is Lz

# Given Lz and stretching factor σ > 1 the top cell height is Δzₜₒₚ = Lz * (σ - 1) / σ^(Nz - 1)

grid = RectilinearGrid(arch,
                       topology = (Periodic, Bounded, Bounded),
                       size = (Nx, Ny, Nz),
                       halo = (3, 3, 3),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (-Lz, 0)) 

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

α  = 2e-4         # [m s⁻²K⁻¹] thermal expansion coefficient 
g  = 10           # [m s⁻²] gravitational constant
const f = 1e-2    # [s⁻¹]
rho_0   = 1000
parameters = (
              Lx = Lx,
              Ly = Ly,
              Lz = Lz,
              τ = 20. / rho_0,          # surface kinematic wind stress [m² s⁻²]
              bs = -10 * α * g,         # surface vertical buoyancy flux [s⁻²]
              Δb =  20 * α * g,         # surface horizontal buoyancy flux [s⁻²]
              H = Lz,                   # domain depth [m]
              h = 500.0,                # exponential decay scale of stable stratification [m]
              y_sponge = 19/20  * Ly,   # southern boundary of sponge layer [m]
              z_sponge = 1/100 * Lz,    # bottom boundary of top layer [m]
              y_basin  = 1/2 * Ly,           # cutoff of velocity
              λt = (Lz / g)^(1/2) * 10,      # relaxation time scale [s]
              μ = 1 / (Lz / g)^(1/2) / 1000,   # bottom drag damping time-scale [s⁻¹]
              )



@inline mask_y(y, p) = y < p.y_basin ? 1 : 0

@inline function u_stress(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return - p.τ * sin(π * y / p.y_basin) * mask_y(y, p)
end

u_stress_bc = FluxBoundaryCondition(u_stress, discrete_form=true, parameters=parameters)

@inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds - p.μ * p.Lz * model_fields.u[i, j, 1] 
@inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds - p.μ * p.Lz * model_fields.v[i, j, 1]

u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form=true, parameters=parameters)
v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form=true, parameters=parameters)

u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
v_bcs = FieldBoundaryConditions(bottom = v_drag_bc)



#####
##### Coriolis
#####
coriolis = FPlane(f = f)

#####
##### Forcing and initial condition
#####

@inline linear_buoyancy(y, p) =  p.bs + p.Δb/p.Ly * y 

@inline mask_north(y, p) = max(0.0, y - p.y_sponge) / (p.Ly - p.y_sponge)
@inline mask_top(z, p)   = max(0.0, z + p.z_sponge) / (p.z_sponge)

@inline function buoyancy_relaxation(i, j, k, grid, clock, model_fields, p)
    timescale = p.λt
    y = ynode(Center(), j, grid)
    z = znode(Center(), k, grid)
    surface_b = linear_buoyancy(y, p) 
    b = @inbounds model_fields.b[i, j, k]
    return - 1 / timescale * mask_top(z, p) * (b - surface_b)
end

Fb = Forcing(buoyancy_relaxation, discrete_form = true, parameters = parameters)

# Turbulence closures

κz = 0.00001
κh = 0.00001 * (grid.Δxᶜᵃᵃ/grid.Δzᵃᵃᶜ)^2  # [m²/s] horizontal diffusivity
νz = 0.00001 
νh = 0.00001 * (grid.Δxᶜᵃᵃ/grid.Δzᵃᵃᶜ)^2  # [m²/s] horizontal viscosity

horizontal_diffusivity = AnisotropicDiffusivity(νh=νh, νz=νz, κh=κh, κz=κz)

#####
##### Model building
#####

@info "Buiding the immersed boundary..."

solid_wall(x, y, z) = (x - Lx/2  < Lx/50 ) && (x - Lx/2 > - Lx/50) && ( y > Ly/2)

immersed_grid = ImmersedBoundaryGrid(grid, GridFittedBoundary(solid_wall))

@info "Building a model..."

model = NonhydrostaticModel(        grid = immersed_grid,
                               advection = WENO5(),
                                buoyancy = BuoyancyTracer(),
                                coriolis = coriolis,
                                 closure = horizontal_diffusivity,
                                 tracers = :b,
                             timestepper = :RungeKutta3,
                     boundary_conditions = (u=u_bcs, v=v_bcs),
				 forcing = (;b = Fb))


@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
ε(σ) = σ * randn()
bᵢ(x, y, z) = -10 * α * g  +  20 * α * g * y / Ly
uᵢ(x, y, z) = rand()*0.01
vᵢ(x, y, z) = rand()*0.01 
wᵢ(x, y, z) = ε(1e-8)

set!(model, b=bᵢ, u=uᵢ, v=vᵢ, w=wᵢ)

#####
##### Simulation building

wizard = TimeStepWizard(cfl=0.5, max_change=1.1, max_Δt=60minutes)

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

simulation = Simulation(model, Δt=0.1, stop_iteration=stop_time)
simulation.callbacks[:wizard]   = Callback(wizard, IterationInterval(100))
simulation.callbacks[:progress] = Callback(print_progress, IterationInterval(100))
#####
##### Diagnostics
#####

u, v, w = model.velocities
b = model.tracers.b

ζz = ComputedField(∂x(v) - ∂y(u))
ζy = ComputedField(∂z(u) - ∂x(w))
ζx = ComputedField(∂y(w) - ∂z(v))

bx = ComputedField(∂x(b)) 
by = ComputedField(∂y(b)) 
bz = ComputedField(∂z(b)) 


outputs = (; b, bx, by, bz, ζx, ζy, ζz, u, v, w)

#####
##### Build checkpointer and output writer
#####

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(checkpointer_time),
                                                        prefix = "basin",
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
                                                       prefix = "basin_$(side)_slice",
                                                       force = true)
end

 @info "Running the simulation..."

try
    run!(simulation, pickup=false)
catch err
    @info "run! threw an error! The error message is"
    showerror(stdout, err)
end


