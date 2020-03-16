module FreeSlip

using Statistics, Printf, JLD2, Glob

using Oceananigans, Oceananigans.Forcing, Oceananigans.BoundaryConditions, Oceananigans.OutputWriters,
        Oceananigans.Fields

import Oceananigans: RegularCartesianGrid

# Functions that define the forced flow problem

 ξ(t) = 1 + sin(t^2)
ξ′(t) = 2t * cos(t^2)

 f(x, t) =   cos(x - ξ(t))
fₓ(x, t) = - sin(x - ξ(t))

Fᵘ(x, y, z, t) = (4 * f(x, t) - 2 * ξ′(t) * fₓ(x, t)) * cos(z)
Fʷ(x, y, z, t) = sin(2z) / 2

u(x, y, z, t) = f(x, t) * cos(z)
w(x, y, z, t) = -fₓ(x, t) * sin(z)

function setup_free_slip_simulation(; Nx, Nz, CFL, architecture=CPU())

    grid = RegularCartesianGrid(size=(Nx, 1, Nz), x=(0, 2π), y=(0, 1), z=(0, π), 
                                topology=(Periodic, Periodic, Bounded))

    model = IncompressibleModel(architecture = architecture,
                                        grid = grid,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = nothing,
                                     closure = ConstantIsotropicDiffusivity(ν=1),
                                     forcing = ModelForcing(u=SimpleForcing(Fᵘ), w=SimpleForcing(Fʷ)))

    set!(model, u = (x, y, z) -> u(x, y, z, 0), 
                w = (x, y, z) -> w(x, y, z, 0))

    h = min(2π/Nx, 1/Nz)
    Δt = h * CFL # Velocity scale = 1

    simulation = Simulation(model, Δt=Δt, stop_time=π, progress_frequency=1)

    prefix = @sprintf("forced_free_slip_Nx%d_Nz%d_CFL%.0e", Nx, Nz, CFL)
    simulation.output_writers[:fields] = JLD2OutputWriter(model, FieldOutputs(model.velocities);
                                                          interval=π/16, prefix=prefix, force=true)

    return simulation
end

function setup_and_run_free_slip_simulation(; setup...)
    simulation = setup_free_slip_simulation(; setup...)
    println("Running free slip simulation with Nx = $(setup[:Nx]), Nz = $(setup[:Nz]), CFL = $(setup[:CFL])")
    @time run!(simulation)
    return nothing
end

function setup_and_run_free_slip_simulations(setups...)
    for setup in setups
        setup_and_run_free_slip_simulation(; setup...)
    end
    return nothing
end

end # module
