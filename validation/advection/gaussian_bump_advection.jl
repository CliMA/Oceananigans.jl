using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans.ImmersedBoundaries
using Oceananigans.Grids: znode
using Oceananigans.Advection: VelocityStencil, VorticityStencil

arch = GPU()

Nx = 120
Ny = 42
Nz = 40

H  = 4500.0
L  = 25.0e3
Δh = 0.9H
dx = 5.0e3
Lx = dx*Nx
Ly = dx*Ny  

f  = 1e-4
N  = 1.5*f*L/H 

@inline gaussian_bump(x, y) = - H + Δh * exp( - (x^2 + y^2) / (2*L^2)) 

grid = RectilinearGrid(arch, size = (Nx, Ny, Nz), halo = (4, 4, 4), 
                       x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = (-H, 0), 
                       topology = (Periodic, Periodic, Bounded))

ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(gaussian_bump))

parameters = (ΔB = 0.04,
              h  = 1000.0,
              Lz = H,
              u₀ = 0.25,
              v₀ = 0.0,
              λ = 1/(1hours),
              Nx = Nx,
              bounds = (Nx ÷ 8))

@inline initial_buoyancy(z, p) = p.ΔB * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))

@inline mask(i, p) = i < p.bounds ? (p.bounds - i) / (p.bounds - 1) : i > p.Nx - p.bounds ? (i - p.Nx + p.bounds) / p.bounds : 0.0

@inline velocity_restoring_v(i, j, k, grid, clock, fields, p) = - p.λ * (fields.v[i, j, k] - p.v₀) * mask(i, p)
@inline velocity_restoring_u(i, j, k, grid, clock, fields, p) = - p.λ * (fields.u[i, j, k] - p.u₀) * mask(i, p)

@inline function buoyancy_restoring_b(i, j, k, grid, clock, fields, p)
    z = znode(Center(), k, grid)
    target_b = initial_buoyancy(z, p)
    b = @inbounds fields.b[i, j, k]

    return - p.λ * (b - target_b) * mask(i, p) 
end

u_forcing =  Forcing(velocity_restoring_u; discrete_form=true, parameters)
v_forcing =  Forcing(velocity_restoring_v; discrete_form=true, parameters)
b_forcing =  Forcing(buoyancy_restoring_b; discrete_form=true, parameters)

buoyancy = BuoyancyTracer()

# Quadratic bottom drag:
μ = 0.003 # ms⁻¹

@inline speed(i, j, k, grid, fields) = (fields.u[i, j, k]^2 + fields.v[i, j, k]^2)^0.5

@inline u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, 1] * speed(i, j, 1, grid, fields)
@inline v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, 1] * speed(i, j, 1, grid, fields)

@inline u_immersed_bottom_drag(i, j, k, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, k] * speed(i, j, k, grid, fields) 
@inline v_immersed_bottom_drag(i, j, k, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, k] * speed(i, j, k, grid, fields) 

drag_u = FluxBoundaryCondition(u_immersed_bottom_drag, discrete_form=true, parameters = μ)
drag_v = FluxBoundaryCondition(v_immersed_bottom_drag, discrete_form=true, parameters = μ)

u_immersed_bc = ImmersedBoundaryCondition(bottom = drag_u)
v_immersed_bc = ImmersedBoundaryCondition(bottom = drag_v)

u_bottom_drag_bc = FluxBoundaryCondition(u_bottom_drag, discrete_form = true, parameters = μ)
v_bottom_drag_bc = FluxBoundaryCondition(v_bottom_drag, discrete_form = true, parameters = μ)

u_bcs = FieldBoundaryConditions(bottom = u_bottom_drag_bc, immersed = u_immersed_bc)
v_bcs = FieldBoundaryConditions(bottom = u_bottom_drag_bc, immersed = v_immersed_bc)

model = HydrostaticFreeSurfaceModel(; grid = ibg,
                                    buoyancy, coriolis = FPlane(; f),
                                    free_surface = ImplicitFreeSurface(),
                                    tracers = :b, 
                                    tracer_advection = WENOFifthOrder(nothing),
                                    forcing = (; u = u_forcing, v = v_forcing, b = b_forcing),
                                    boundary_conditions = (u = u_bcs, v = v_bcs),
                                    momentum_advection = WENOFifthOrder(nothing, vector_invariant = VelocityStencil()))

g  = model.free_surface.gravitational_acceleration
b = model.tracers.b
u, v, w = model.velocities
set!(b, (x, y, z) -> initial_buoyancy(z, parameters))

wave_speed = sqrt(g * H)
Δt = min(10minutes, 10*dx / wave_speed)

simulation = Simulation(model, Δt = Δt, stop_time = 10days)

progress(sim) = @info "time $(prettytime(sim.model.clock.time)), maximum u: $(maximum(sim.model.velocities.u)), maximum v: $(maximum(sim.model.velocities.v))"

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))

run!(simulation)