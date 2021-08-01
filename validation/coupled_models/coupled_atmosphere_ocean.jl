using Oceananigans
using Oceananigans.Units
using Printf
using GLMakie

#####
##### Utilities
#####

using Oceananigans.Models: AbstractModel

import Oceananigans.TimeSteppers: time_step!, update_state!
import Oceananigans: fields

const ρ_atmos = 1 # kg m⁻³
const ρ_ocean = 1024 # kg m⁻³

struct CoupledAtmosphereOceanModel{O, A, C} <: AbstractModel{Nothing}
    atmos :: A
    ocean :: O
    clock :: C
end

function CoupledAtmosphereOceanModel(atmos, ocean)
    clock = atmos.clock
    return CoupledAtmosphereOceanModel(atmos, ocean, clock)
end

fields(model::CoupledAtmosphereOceanModel) = fields(model.atmos) # convenience hack for now

function update_state!(coupled_model::CoupledAtmosphereOceanModel, update_atmos_ocean_state=true)
    atmos_model = coupled_model.atmos
    ocean_model = coupled_model.ocean

    if update_atmos_ocean_state
        update_state!(ocean_model)
        update_state!(atmos_model)
    end

    uo, vo, wo = ocean_model.velocities
    ua, va, wa = atmos_model.velocities
    atmos_grid = atmos_model.grid
    atmos_surface_flux_u = ua.boundary_conditions.bottom.condition
    atmos_surface_flux_v = va.boundary_conditions.bottom.condition
    ocean_surface_flux_u = uo.boundary_conditions.top.condition
    ocean_surface_flux_v = vo.boundary_conditions.top.condition

    # Make this cleaner...
    Nx, Ny, Nz = size(atmos_grid)
    Hx, Hy, Hz = atmos_grid.Hx, atmos_grid.Hy, atmos_grid.Hz
    ii = Hx+1:Hx+Nx
    jj = 1 # Hy+1:Hy+Ny
    k = atmos_grid.Hz+1
    ua₁ = view(parent(ua), ii, jj, k:k)
    va₁ = view(parent(va), ii, jj, k:k)

    cᴰ = 2e-3
    @. atmos_surface_flux_u = - cᴰ * ua₁ * sqrt(ua₁^2 + va₁^2)
    @. atmos_surface_flux_v = - cᴰ * va₁ * sqrt(ua₁^2 + va₁^2)

    @. ocean_surface_flux_u = ρ_atmos / ρ_ocean * atmos_surface_flux_u
    @. ocean_surface_flux_v = ρ_atmos / ρ_ocean * atmos_surface_flux_v

    return nothing
end

function time_step!(coupled_model::CoupledAtmosphereOceanModel, Δt; euler=false)
    time_step!(coupled_model.ocean, Δt; euler)
    time_step!(coupled_model.atmos, Δt; euler)
    update_state!(coupled_model, false)
    return nothing
end

#####
##### Common settings
#####

arch = CPU()
Nx = 128
Nz = 128

Lx = 1kilometer

Lz_atmos = 1kilometer
Lz_ocean = 100 # meters

coriolis = FPlane(latitude=45)

#####
##### Atmos model setup
#####

U₀ = 10 # initial and top velocity

atmos_grid = RegularRectilinearGrid(size = (Nx, Nz),
                                    x = (0, Lx),
                                    z = (0, Lz_atmos),
                                    topology = (Periodic, Flat, Bounded))

# Store boundary fluxes in arrays
atmos_surface_flux_u = arch isa CPU ? zeros(atmos_grid.Nx, atmos_grid.Ny) : CUDA.zeros(atmos_grid.Nx, atmos_grid.Ny)
atmos_surface_flux_v = arch isa CPU ? zeros(atmos_grid.Nx, atmos_grid.Ny) : CUDA.zeros(atmos_grid.Nx, atmos_grid.Ny)

# Atmos velocity boundary conditions
atmos_u_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(atmos_surface_flux_u),
                                      top = ValueBoundaryCondition(U₀))

atmos_v_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(atmos_surface_flux_v))

atmos_equation_of_state = LinearEquationOfState(α=2e-4)
     
atmos_model = NonhydrostaticModel(architecture = arch,
                                  grid = atmos_grid,
                                  timestepper = :RungeKutta3,
                                  advection = UpwindBiasedFifthOrder(),
                                  boundary_conditions = (u=atmos_u_bcs, v=atmos_v_bcs),
                                  closure = IsotropicDiffusivity(ν=1e-2, κ=1e-2),
                                  coriolis = coriolis,
                                  tracers = :T,
                                  buoyancy = SeawaterBuoyancy(constant_salinity=true,
                                                              equation_of_state = atmos_equation_of_state))

uᵢ(x, y, z) = U₀ * (1 + 1e-3 * randn())
set!(atmos_model, u=uᵢ)

#####
##### Ocean model setup
#####

ocean_grid = RegularRectilinearGrid(size = (Nx, Nz),
                                    x = (0, Lx),
                                    z = (-Lz_ocean, 0),
                                    topology = (Periodic, Flat, Bounded))

# Make sure horizontal grids are the same
@assert ocean_grid.Nx == atmos_grid.Nx
@assert ocean_grid.Ny == atmos_grid.Ny
@assert ocean_grid.Lx == atmos_grid.Lx
@assert ocean_grid.Ly == atmos_grid.Ly

# Ocean velocity boundary conditions
ocean_surface_flux_u = arch isa CPU ? zeros(atmos_grid.Nx, atmos_grid.Ny) : CUDA.zeros(atmos_grid.Nx, atmos_grid.Ny)
ocean_surface_flux_v = arch isa CPU ? zeros(atmos_grid.Nx, atmos_grid.Ny) : CUDA.zeros(atmos_grid.Nx, atmos_grid.Ny)

ocean_u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(ocean_surface_flux_u))
ocean_v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(ocean_surface_flux_v))

ocean_equation_of_state = LinearEquationOfState(α=2e-4)
    
ocean_model = NonhydrostaticModel(architecture = arch,
                                  grid = ocean_grid,
                                  timestepper = :RungeKutta3,
                                  advection = UpwindBiasedFifthOrder(),
                                  boundary_conditions = (u=ocean_u_bcs, v=ocean_v_bcs),
                                  closure = IsotropicDiffusivity(ν=1e-3, κ=1e-3),
                                  coriolis = coriolis,
                                  tracers = (:T, :S),
                                  buoyancy = SeawaterBuoyancy(equation_of_state = ocean_equation_of_state))

#####
##### Coupled model + simulation!
#####

coupled_model = CoupledAtmosphereOceanModel(atmos_model, ocean_model)

simulation = Simulation(coupled_model, Δt=1, stop_time=12hours)

function print_progress(sim)
    uo, vo, wo = sim.model.ocean.velocities
    ua, va, wa = sim.model.atmos.velocities
    iter = sim.model.clock.iteration
    time = sim.model.clock.time

    msg = @sprintf("Iter: %d, time: % 20s, uo: (%.3e, %.3e), wo: (%.3e, %.3e), ua: (%.3e, %.3e), wa: (%.3e, %.3e)",
                   iter, prettytime(time),
                   minimum(uo), maximum(uo),
                   minimum(wo), maximum(wo),
                   minimum(ua), maximum(ua),
                   minimum(wa), maximum(wa))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(print_progress, schedule=IterationInterval(10))

simulation.output_writers[:atmos] = JLD2OutputWriter(atmos_model, merge(atmos_model.velocities, atmos_model.tracers),
                                                     schedule = TimeInterval(10minutes),
                                                     prefix = "coupled_model_atmos",
                                                     force = true,
                                                     field_slicer = nothing)

simulation.output_writers[:ocean] = JLD2OutputWriter(ocean_model, merge(ocean_model.velocities, ocean_model.tracers),
                                                     schedule = TimeInterval(10minutes),
                                                     prefix = "coupled_model_ocean",
                                                     force = true,
                                                     field_slicer = nothing)

run!(simulation)

######
###### Visualize
######

u_ocean = FieldTimeSeries("coupled_model_ocean.jld2", "u")
u_atmos = FieldTimeSeries("coupled_model_atmos.jld2", "u")
w_ocean = FieldTimeSeries("coupled_model_ocean.jld2", "w")
w_atmos = FieldTimeSeries("coupled_model_atmos.jld2", "w")

fig = Figure(backgroundcolor = RGBf0(0.98, 0.98, 0.98), resolution = (2000, 1000))
u_axs = fig[1:2, 1] = [Axis(fig, title = t) for t in ["Atmos u", "Ocean u"]]
w_axs = fig[1:2, 2] = [Axis(fig, title = t) for t in ["Atmos w", "Ocean w"]]
ax_u_atmos, ax_u_ocean = u_axs
ax_w_atmos, ax_w_ocean = w_axs

n = Node(1) 

ua = @lift interior(u_atmos[$n])[:, 1, :]
uo = @lift interior(u_ocean[$n])[:, 1, :]

wa = @lift interior(w_atmos[$n])[:, 1, :]
wo = @lift interior(w_ocean[$n])[:, 1, :]

heatmap!(ax_u_atmos, ua)
heatmap!(ax_u_ocean, uo)

heatmap!(ax_w_atmos, wa)
heatmap!(ax_w_ocean, wo)

record(fig, "coupled_model.mp4", 1:length(u_ocean.times); framerate = 8) do save_point
    n[] = save_point
end

display(fig)

