module ForcedFlowFreeSlip

using Printf

using Oceananigans, Oceananigans.Forcing, Oceananigans.BoundaryConditions, Oceananigans.OutputWriters,
        Oceananigans.Fields

# Functions that define the forced flow problem

 ξ(t) = 1 + sin(t^2)
ξ′(t) = 2t * cos(t^2)

 f(x, t) =   cos(x - ξ(t))
fₓ(x, t) = - sin(x - ξ(t))

Fᵘ(x, y, t) = (4 * f(x, t) - 2 * ξ′(t) * fₓ(x, t)) * cos(y)
Fᵛ(x, y, t) = sin(2y) / 2

u(x, y, t) =   f(x, t) * cos(y)
v(x, y, t) = -fₓ(x, t) * sin(y)

#####
##### x, z
#####

function setup_xz_simulation(; Nx, Δt, stop_iteration, architecture=CPU(), dir="data")

    grid = RegularCartesianGrid(size=(Nx, 1, Nx), x=(0, 2π), y=(0, 1), z=(0, π), 
                                topology=(Periodic, Periodic, Bounded))

    model = IncompressibleModel(architecture = architecture,
                                        grid = grid,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = nothing,
                                     closure = ConstantIsotropicDiffusivity(ν=1),
                                     forcing = ModelForcing(u = SimpleForcing((x, y, z, t) -> Fᵘ(x, z, t)), 
                                                            w = SimpleForcing((x, y, z, t) -> Fᵛ(x, z, t)))
                                )

    set!(model, u = (x, y, z) -> u(x, z, 0), 
                w = (x, y, z) -> v(x, z, 0))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, progress_frequency=stop_iteration)

    simulation.output_writers[:fields] = JLD2OutputWriter(model, FieldOutputs(model.velocities);
                                                          dir = dir, force = true,
                                                          prefix = @sprintf("forced_free_slip_xz_Nx%d_Δt%.1e", Nx, Δt),
                                                          interval = stop_iteration * Δt / 2)

    return simulation
end

function setup_and_run_xz(; setup...)
    simulation = setup_xz_simulation(; setup...)
    println("Running free slip simulation in x, z with Nx = $(setup[:Nx]), Δt = $(setup[:Δt])")
    @time run!(simulation)
    return nothing
end

#####
##### x, y
#####

function setup_xy_simulation(; Nx, Δt, stop_iteration, architecture=CPU(), dir="data")

    grid = RegularCartesianGrid(size=(Nx, Nx, 1), x=(0, 2π), y=(0, π), z=(0, 1), 
                                topology=(Periodic, Bounded, Bounded))

    model = IncompressibleModel(architecture = architecture,
                                        grid = grid,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = nothing,
                                     closure = ConstantIsotropicDiffusivity(ν=1),
                                     forcing = ModelForcing(u = SimpleForcing((x, y, z, t) -> Fᵘ(x, y, t)), 
                                                            v = SimpleForcing((x, y, z, t) -> Fᵛ(x, y, t)))
                                )

    set!(model, u = (x, y, z) -> u(x, y, 0), 
                v = (x, y, z) -> v(x, y, 0))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, progress_frequency=stop_iteration)

    simulation.output_writers[:fields] = JLD2OutputWriter(model, FieldOutputs(model.velocities);
                                                          dir = dir, force = true,
                                                          prefix = @sprintf("forced_free_slip_xy_Nx%d_Δt%.1e", Nx, Δt),
                                                          interval=stop_iteration * Δt / 2)

    return simulation
end

function setup_and_run_xy(; setup...)
    simulation = setup_xy_simulation(; setup...)
    println("Running free slip simulation in x, y with Nx = $(setup[:Nx]), Δt = $(setup[:Δt])")
    @time run!(simulation)
    return nothing
end

end # module
