using Oceananigans
using Oceananigans.Fields: VelocityFields
using Oceananigans.Models.HydrostaticFreeSurfaceModels: PrescribedVelocityFields
using Printf

import Base

# the initial condition
@inline G(x, β, z) = exp(-β*(x - z)^2)
@inline F(x, α, a) = √(max(1 - α^2*(x-a)^2, 0.0))

const Z = -0.7
const δ = 0.005
const β = log(2)/(36*δ^2)
const a = 0.5
const α = 10

@inline function bᵢ(x) 
    if x <= -0.6 && x >= -0.8
        return 1/6*(G(x, β, Z-δ) + 4*G(x, β, Z) + G(x, β, Z+δ))
    elseif x <= -0.2 && x >= -0.4
        return 1.0
    elseif x <= 0.2 && x >= 0.0
        return 1.0 - abs(10 * (x - 0.1))
    elseif x <= 0.6 && x >= 0.4
        return 1/6*(F(x, α, a-δ) + 4*F(x, α, a) + F(x, α, a+δ))
    else
        return 0.0
    end
end

function one_dimensional_advection(N, advection = WENO(order = 7), CFL = 0.75; timestepper = :QuasiAdamsBashforth2)

    grid = RectilinearGrid(size = N, halo = 6, x = (-1, 1), topology = (Periodic, Flat, Flat))
    
    velocities = PrescribedVelocityFields(; u = 1)

    model = HydrostaticFreeSurfaceModel(; grid, tracers = :b, timestepper, tracer_advection = advection, velocities) 

    set!(model.tracers.b, bᵢ)

    Δt = CFL * minimum_xspacing(grid)

    simulation  = Simulation(model; Δt, stop_time = 2)

    return simulation
end

function one_dimensional_advection_nhydrostatic(N, advection = WENO(order = 7); timestepper = :RungeKutta3)

    grid = RectilinearGrid(size = N, halo = 6, x = (-1, 1), topology = (Periodic, Flat, Flat))
    
    model = NonhydrostaticModel(; grid, tracers = :b, timestepper, advection) 

    set!(model.tracers.b, bᵢ)
    set!(model.velocities.u, 1.0)

    Δt = 0.2 * minimum_xspacing(grid)

    simulation  = Simulation(model; Δt, stop_time = 2)

    return simulation
end
