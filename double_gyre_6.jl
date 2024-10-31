using Oceananigans
using Oceananigans.Units
using Oceananigans.Operators
using Oceananigans.Grids: φnode
using Oceananigans.AbstractOperations: GridMetricOperation
using Printf


data_directory = "/nobackup1/sandre/OceananigansData/"
arch = GPU()
Nz = 2
Nxy = 32 * 8
Lz = 1800
σ = 1.3

z_faces(k) = -Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ));
z_faces_2 = ZStarVerticalCoordinate(z_faces)

grid = LatitudeLongitudeGrid(arch; size=(Nxy, Nxy, Nz),
    latitude=(15, 75),
    longitude=(0, 60),
    halo=(5, 5, 5),
    z=z_faces_2)

#####
##### Parameters
#####

θ⁺ = 30      # ᵒC maximum temperature
θ⁻ = 0       # ᵒC maximum temperature
α = 2e-4    # ᵒC⁻¹ thermal expansion coefficient
ρ₀ = 1000    # kg m⁻³ reference density
g = 9.80665 # m s⁻² gravitational acceleration
λ = 30days  # time scale for restoring 

#####
##### Numerics
#####

Δt = 40minutes * (32 / Nxy)

Δx = minimum_xspacing(grid)
Δy = minimum_yspacing(grid)

Δs = sqrt(1 / (1 / Δx^2 + 1 / Δy^2))
sp = sqrt(g * grid.Lz)
CFL = 0.75
Δτ = Δs / sp * CFL

substeps = ceil(Int, 2 * Δt / Δτ)

coriolis = HydrostaticSphericalCoriolis()
momentum_advection = WENOVectorInvariant(vorticity_order=5)
tracer_advection = WENO(order=5)
free_surface = SplitExplicitFreeSurface(grid; substeps)

numerics = (; coriolis, free_surface, momentum_advection, tracer_advection)

#####
##### Closure
#####

closure1 = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=1.0,
    background_κz=1e-5,
    convective_νz=1e-2,
    background_νz=1e-2)
closure2 = HorizontalScalarDiffusivity(ν=10^3, κ=10^3)
closure = (closure1, closure2)

##### 
##### Boundary Conditions
#####

@inline function wind_stress(i, j, grid, clock, fields, p)
    τ₀ = p.τ₀
    y = (φnode(j, grid, Center()) - p.φ₀) / grid.Ly

    return τ₀ * cos(2π * y)
end

@inline function buoyancy_restoring(i, j, grid, clock, fields, p)
    b = @inbounds fields.b[i, j, grid.Nz]
    y = (φnode(j, grid, Center()) - p.φ₀) / grid.Ly
    b★ = p.Δb * (1 - y)

    return p.𝓋 * (b - b★)
end

Δz₀ = 50.0 # minimum([20.0, Δzᶜᶜᶜ(1, 1, grid.Nz, grid)]) # Surface layer thickness

Δb = α * g * (θ⁺ - θ⁻) # Buoyancy difference

parameters = (; τ₀=0.1 / ρ₀, # Wind stress 
    φ₀=15,       # Latitude of southern edge
    Δb,            # Buoyancy difference
    𝓋=Δz₀ / λ)  # Pumping velocity for restoring

u_boundary = FluxBoundaryCondition(wind_stress; discrete_form=true, parameters)
b_boundary = FluxBoundaryCondition(buoyancy_restoring; discrete_form=true, parameters)

@inline quadratic_drag_u(x, y, t, u, v, drag_coeff) = -drag_coeff * u * sqrt(u^2 + v^2)
u_bottom_bc = FluxBoundaryCondition(quadratic_drag_u, field_dependencies=(:u, :v), parameters=1e-3)
@inline quadratic_drag_v(x, y, t, u, v, drag_coeff) = -drag_coeff * v * sqrt(u^2 + v^2)
v_bottom_bc = FluxBoundaryCondition(quadratic_drag_v, field_dependencies=(:u, :v), parameters=1e-3)

no_slip = ValueBoundaryCondition(0.0)

u_bcs = FieldBoundaryConditions(north=no_slip, south=no_slip, top=u_boundary, bottom=u_bottom_bc)
v_bcs = FieldBoundaryConditions(west=no_slip, east=no_slip, bottom=v_bottom_bc)
b_bcs = FieldBoundaryConditions(top=b_boundary)

#####  
##### Model
#####

model = HydrostaticFreeSurfaceModel(; grid,
    boundary_conditions=(u=u_bcs, v=v_bcs, b=b_bcs),
    buoyancy=BuoyancyTracer(),
    tracers=:b,
    numerics...,
    closure)

N² = Δb / grid.Lz
bᵢ(x, y, z) = N² * (grid.Lz + z)

set!(model, b=bᵢ)

#####
##### Simulation
#####

simulation = Simulation(model; Δt, stop_time=600000days)

#####
##### Output
#####

Δzᶜᶜ = GridMetricOperation((Center, Center, Center), Oceananigans.AbstractOperations.Δz, model.grid)
Δzᶠᶜ = GridMetricOperation((Face, Center, Center), Oceananigans.AbstractOperations.Δz, model.grid)
Δzᶜᶠ = GridMetricOperation((Center, Face, Center), Oceananigans.AbstractOperations.Δz, model.grid)

field_outputs = merge(model.velocities,
    model.tracers,
    model.pressure,
    (; Δzᶜᶜ, Δzᶠᶜ, Δzᶜᶠ))

function progress(sim)
    w = interior(sim.model.velocities.w, :, :, sim.model.grid.Nz + 1)
    u = sim.model.velocities.u
    b = sim.model.tracers.b

    msg0 = @sprintf("Time: %s iteration %d ", prettytime(sim.model.clock.time), sim.model.clock.iteration)
    msg1 = @sprintf("extrema w: %.2e %.2e ", maximum(w), minimum(w))
    msg2 = @sprintf("extrema u: %.2e %.2e ", maximum(u), minimum(u))
    msg3 = @sprintf("extrema b: %.2e %.2e ", maximum(b), minimum(b))
    msg4 = @sprintf("extrema Δz: %.2e %.2e ", maximum(Δzᶜᶜ), minimum(Δzᶜᶜ))
    @info msg0 * msg1 * msg2 * msg3 * msg4

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

simulation.output_writers[:snapshots] = JLD2OutputWriter(model, field_outputs,
    overwrite_existing=true,
    schedule=TimeInterval(30days),
    filename=data_directory * "baroclinic_double_gyre_6")

simulation.output_writers[:free_surface] = JLD2OutputWriter(model, (; η=model.free_surface.η),
    overwrite_existing=true,
    indices=(:, :, grid.Nz + 1),
    schedule=TimeInterval(30days),
    filename=data_directory * "baroclinic_double_gyre_free_surface_6")

run!(simulation)