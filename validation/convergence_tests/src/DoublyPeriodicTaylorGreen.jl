module DoublyPeriodicTaylorGreen

using Printf

using Oceananigans
using Oceananigans.OutputWriters

# Advected vortex: ψ(x, y, t) = exp(-2t) * cos(x - U*t) * cos(y)
u(x, y, t, U=1) = U + exp(-2t) * cos(x - U*t) * sin(y)
v(x, y, t, U=1) =   - exp(-2t) * sin(x - U*t) * cos(y)

#####
##### x, y
#####

const DATA_DIR = joinpath(@__DIR__, "..", "data")

function setup_simulation(; Nx, Δt, stop_iteration, U=1, architecture=CPU(), dir=DATA_DIR)

    grid = RectilinearGrid(size=(Nx, Nx, 1), x=(0, 2π), y=(0, 2π), z=(0, 1),
                                topology=(Periodic, Periodic, Bounded))

    model = NonhydrostaticModel(architecture = architecture,
                                        grid = grid,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = nothing,
                                     closure = IsotropicDiffusivity(ν=1))

    set!(model, u = (x, y, z) -> u(x, y, 0, U),
                v = (x, y, z) -> v(x, y, 0, U))

    function print_progress(simulation)
        model = simulation.model
        i, t = model.clock.iteration, model.clock.time
        progress = 100 * (i / simulation.stop_iteration)
        @info @sprintf("[%05.2f%%] iteration: %d, time: %.5e", progress, i, t)
        return nothing
    end

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, progress=print_progress, iteration_interval=125)

    simulation.output_writers[:fields] = JLD2OutputWriter(model, model.velocities;
                                                          dir = dir, force = true, field_slicer = nothing,
                                                          prefix = @sprintf("taylor_green_Nx%d_Δt%.1e", Nx, Δt),
                                                          schedule = TimeInterval(stop_iteration * Δt / 10))

    return simulation
end

function setup_and_run(; setup...)

    simulation = setup_simulation(; setup...)

    @info "Running decaying Taylor-Green vortex simulation with Nx = Ny = $(setup[:Nx]), Δt = $(setup[:Δt])"

    @time run!(simulation)
    return nothing
end

end # module
