module TwoDimensionalDiffusion

using Printf

using Oceananigans, Oceananigans.OutputWriters

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
                          topo=(Periodic, Periodic, Bounded))

    grid = RegularCartesianGrid(size=(Nx, Nx, 1), x=(0, Lx(topo)), y=(0, Ly(topo)), z=(0, 1), topology=topo)

    model = IncompressibleModel(architecture = architecture,
                                        grid = grid,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = :c,
                                     closure = ConstantIsotropicDiffusivity(κ=1))

    set!(model, c = (x, y, z) -> c(x, y, 0)) 

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, progress_frequency=stop_iteration)

    simulation.output_writers[:fields] = 
        JLD2OutputWriter(model, FieldOutputs(model.tracers); dir = dir, force = true, 
                         prefix = @sprintf("%s_%s_diffusion_Nx%d_Δt%.1e", "$(topo[1])", "$(topo[2])", Nx, Δt),
                         interval = stop_iteration * Δt / 100)

    return simulation
end

function run_simulation(; setup...)
    simulation = setup_simulation(; setup...)
    println("Running two dimensional diffusion simulation in x, y with Nx = $(setup[:Nx]), Δt = $(setup[:Δt])")
    @time run!(simulation)
    return nothing
end

end # module
