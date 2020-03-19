module DoublyPeriodicDiffusion

using Printf

using Oceananigans, Oceananigans.OutputWriters

# 2D diffusing sinuosoid
c(x, y, t) = exp(-2t) * cos(x) * cos(y)

Lx(::Tuple{Type{Periodic}, Y}) where Y = 2π
Lx(::Tuple{Type{Bounded},  Y}) where Y = π

Ly(::Tuple{X, Type{Periodic}}) where X = 2π
Ly(::Tuple{X, Type{Bounded}})  where X = π

#####
##### x, y
#####

function setup_simulation(; Nx, Δt, stop_iteration, architecture=CPU(), dir="data",
                          topo=(Periodic, Periodic, Bounded))

    grid = RegularCartesianGrid(size=(Nx, Nx, 1), x=(0, Lx(topo)), y=(0, Ly(topo)), 
                                z=(0, 1), topology=topo)

    model = IncompressibleModel(architecture = architecture,
                                        grid = grid,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = :c,
                                     closure = ConstantIsotropicDiffusivity(ν=1))

    set!(model, c = (x, y, z) -> c(x, y, 0)) 

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, progress_frequency=stop_iteration)

    simulation.output_writers[:fields] = 
        JLD2OutputWriter(model, FieldOutputs(model.velocities); dir = dir, force = true, 
                         prefix = @sprintf("doubly_periodic_diffusion_Nx%d_Δt%.1e", Nx, Δt),
                         interval = stop_iteration * Δt / 100)

    return simulation
end

function run_simulation(; setup...)
    simulation = setup_xy_simulation(; setup...)
    println("Running free decay simulation in x, y with Nx = $(setup[:Nx]), Δt = $(setup[:Δt])")
    @time run!(simulation)
    return nothing
end

end # module
