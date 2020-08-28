module TwoDimensionalDiffusion

using Printf, Statistics

using Oceananigans, Oceananigans.OutputWriters

using Oceananigans.Fields: nodes

include("analysis.jl")

# 2D diffusing sinuosoid
c(x, y, t) = exp(-2t) * cos(x) * cos(y)

instantiate(t) = Tuple(ti() for ti in t)
Lx(topo) = Lx(instantiate(topo))
Ly(topo) = Ly(instantiate(topo))

Lx(::Tuple{Periodic, Y, Z}) where {Y, Z} = 2π
Lx(::Tuple{Bounded,  Y, Z}) where {Y, Z} = π

Ly(::Tuple{X, Periodic, Z}) where {X, Z} = 2π
Ly(::Tuple{X, Bounded,  Z}) where {X, Z} = π

#####
##### x, y
#####

function setup_simulation(; Nx, Δt, stop_iteration, architecture=CPU(), dir="data",
                          topo=(Periodic, Periodic, Bounded), output=false)

    grid = RegularCartesianGrid(size=(Nx, Nx, 1), x=(0, Lx(topo)), y=(0, Ly(topo)), z=(0, 1), topology=topo)

    model = IncompressibleModel(architecture = architecture,
                                        grid = grid,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = :c,
                                     closure = IsotropicDiffusivity(κ=1))

    set!(model, c = (x, y, z) -> c(x, y, 0))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    if output
        simulation.output_writers[:fields] =
            JLD2OutputWriter(model, FieldOutputs(model.tracers); dir = dir, force = true,
                             prefix = @sprintf("%s_%s_diffusion_Nx%d_Δt%.1e", "$(topo[1])", "$(topo[2])", Nx, Δt),
                             time_interval = stop_iteration * Δt / 10)
    end

    return simulation
end

function run_simulation(; setup...)
    simulation = setup_simulation(; setup...)
    println("Running two dimensional diffusion simulation in x, y with Nx = $(setup[:Nx]), Δt = $(setup[:Δt])")
    @time run!(simulation)
    return nothing
end

function run_and_analyze(; setup...)
    simulation = setup_simulation(; setup...)
    println("Running two dimensional diffusion simulation in x, y with Nx = $(setup[:Nx]), Δt = $(setup[:Δt])")
    @time run!(simulation)

    c_simulation = simulation.model.tracers.c

    x, y, z, t = (nodes(c_simulation)..., simulation.model.clock.time)

    c_analytical = c.(x, y, t)

    c_simulation = interior(c_simulation)

    @show size(c_analytical)
    @show size(c_simulation)

    return compute_error(c_simulation, c_analytical)
end

end # module
