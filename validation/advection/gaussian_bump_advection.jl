using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans.ImmersedBoundaries
using Oceananigans.Advection: VelocityStencil, VorticityStencil

arch = GPU()

Nx = 120
Ny = 42
Nz = 40

H  = 4500.0
L  = 25.0e3
Δh = 0.9H
Δx = 5.0e3
Lx = Δx*Nx
Ly = Δx*Ny  

args = ()

f  = 1e-4
N  = 1.5*f*L/H 
bz = N * N

gaussian_bump(x, y) = - H + Δh * exp( - (x^2 + y^2) / (2*L^2)) 
@inline initial_buoyancy(x, y, z) = bz * (z + H)

grid = RectilinearGrid(arch, size = (Nx, Ny, Nz), halo = (4, 4, 4), 
                       x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = (-H, 0), 
                       topology = (Periodic, Periodic, Bounded))

ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(gaussian_bump))

restoring_bounds = Int(Nx / 8)
λ = 1/(1hours)

@inline function velocity_restoring_u(i, j, k, grid, clock, fields, p)
    if i < p.bounds 
        return - p.λ * (fields.u[i, j, k] - p.u₀) * (p.bounds - i) / p.bounds
    elseif i > grid.Nx - p.bounds
        return - p.λ * (fields.u[i, j, k] - p.u₀) * (i - grid.Nx + p.bounds) / p.bounds
    else
        return zero(grid)
    end
end

@inline function velocity_restoring_v(i, j, k, grid, clock, fields, p)
    if i < p.bounds 
        return - p.λ * (fields.v[i, j, k] - p.u₀) * (p.bounds - i) / p.bounds
    elseif i > grid.Nx - p.bounds
        return - p.λ * (fields.v[i, j, k] - p.u₀) * (i - grid.Nx + p.bounds) / p.bounds
    else
        return zero(grid)
    end
end

u_forcing =  Forcing(velocity_restoring_u, discrete_form=true, parameters=(u₀ = 0.25, bounds = restoring_bounds, λ = λ))
v_forcing =  Forcing(velocity_restoring_v, discrete_form=true, parameters=(u₀ = 0.00, bounds = restoring_bounds, λ = λ))

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
                                    tracer_advection = WENO5(nothing),
                                    forcing = (; u = u_forcing), #, v = v_forcing),
                                    boundary_conditions = (u = u_bcs, v = v_bcs),
                                    momentum_advection = WENO5(nothing, vector_invariant = VelocityStencil()))

g  = model.free_surface.gravitational_acceleration
b = model.tracers.b
u, v, w = model.velocities
set!(b, initial_buoyancy)

wave_speed = sqrt(g * H)
Δt = min(10minutes, 10*Δx / wave_speed)

simulation = Simulation(model, Δt = Δt, stop_time = 10days)

progress(sim) = @info "time $(prettytime(sim.model.clock.time)), maximum u: $(maximum(sim.model.velocities.u)), maximum v: $(maximum(sim.model.velocities.v))"

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))

run!(simulation)