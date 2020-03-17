module DoublyPeriodicFreeDecay

using Printf

using Oceananigans, Oceananigans.OutputWriters

# Advected vortex: ψ(x, y, t) = exp(-2t) * cos(x - U*t) * cos(y)
u(x, y, t) = 1 + exp(-2t) * cos(x - t) * sin(y)
v(x, y, t) =   - exp(-2t) * sin(x - t) * cos(y)

#####
##### x, y
#####

function setup_xy_simulation(; Nx, Δt, stop_iteration, architecture=CPU(), dir="data")

    Lx = 2π

    grid = RegularCartesianGrid(size=(Nx, Nx, 1), x=(0, Lx), y=(0, Lx), z=(0, 1), 
                                topology=(Periodic, Periodic, Bounded))

    model = IncompressibleModel(architecture = architecture,
                                        grid = grid,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = nothing,
                                     closure = ConstantIsotropicDiffusivity(ν=1))

    set!(model, u = (x, y, z) -> u(x, y, 0), 
                v = (x, y, z) -> v(x, y, 0))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, progress_frequency=stop_iteration)

    simulation.output_writers[:fields] = JLD2OutputWriter(model, FieldOutputs(model.velocities);
                                                          dir = dir, force = true, 
                                                          prefix = @sprintf("free_decay_xy_Nx%d_Δt%.1e", Nx, Δt),
                                                          interval = stop_iteration * Δt / 2)

    return simulation
end

function setup_and_run_xy(; setup...)
    simulation = setup_xy_simulation(; setup...)
    println("Running free decay simulation in x, y with Nx = $(setup[:Nx]), Δt = $(setup[:Δt])")
    @time run!(simulation)
    return nothing
end

end # module
