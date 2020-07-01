module DoublyPeriodicTaylorGreen

using Printf

using Oceananigans, Oceananigans.OutputWriters

# Advected vortex: ψ(x, y, t) = exp(-2t) * cos(x - U*t) * cos(y)
u(x, y, t, U=1) = U + exp(-2t) * cos(x - U*t) * sin(y)
v(x, y, t, U=1) =   - exp(-2t) * sin(x - U*t) * cos(y)

#####
##### x, y
#####

function setup_simulation(; Nx, Δt, stop_iteration, U=1, architecture=CPU(), dir="data")

    grid = RegularCartesianGrid(size=(Nx, Nx, 1), x=(0, 2π), y=(0, 2π), z=(0, 1), 
                                topology=(Periodic, Periodic, Bounded))

    model = IncompressibleModel(architecture = architecture,
                                        grid = grid,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = nothing,
                                     closure = ConstantIsotropicDiffusivity(ν=1))

    set!(model, u = (x, y, z) -> u(x, y, 0, U), 
                v = (x, y, z) -> v(x, y, 0, U))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, progress_frequency=stop_iteration)

    simulation.output_writers[:fields] = JLD2OutputWriter(model, FieldOutputs(model.velocities);
                                                          dir = dir, force = true, 
                                                          prefix = @sprintf("taylor_green_Nx%d_Δt%.1e", Nx, Δt),
                                                          interval = stop_iteration * Δt / 10)

    return simulation
end

function setup_and_run(; setup...)

    simulation = setup_simulation(; setup...)

    println("""
            Running decaying Taylor-Green vortex simulation in x, y with 

                Nx = $(setup[:Nx]) 
                Δt = $(setup[:Δt])

            """)

    @time run!(simulation)
    return nothing
end

end # module
