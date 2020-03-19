module CosineAdvectionDiffusion

using Printf, Statistics

using Oceananigans, Oceananigans.OutputWriters

include("analysis.jl")

# Advection and diffusion of a cosine.
c(x, y, z, t, U, κ) = exp(-κ * t) * cos(x - U * t)

function run_test(; Nx, Δt, stop_iteration, U = 1, κ = 1e-4,
                  architecture = CPU(), topo = (Periodic, Periodic, Bounded))
                                      
    #####
    ##### Test c and v-advection
    #####

    grid = RegularCartesianGrid(size=(Nx, 1, 1), x=(0, 2π), y=(0, 1), z=(0, 1), topology=topo)

    model = IncompressibleModel(architecture = architecture,
                                        grid = grid,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = :c,
                                     closure = ConstantIsotropicDiffusivity(ν=κ, κ=κ))

    set!(model, u = U, 
                v = (x, y, z) -> c(x, y, z, 0, U, κ),
                c = (x, y, z) -> c(x, y, z, 0, U, κ))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, progress_frequency=stop_iteration)

    println("Running Gaussian advection diffusion test for v and c with Nx = $Nx and Δt = $Δt...")
    run!(simulation)

    # Calculate errors
    c_simulation = model.tracers.c
    c_simulation = interior(c_simulation)[:, 1, 1]

    v_simulation = model.velocities.v
    v_simulation = interior(v_simulation)[:, 1, 1]

    c_analytical = c.(grid.xC, 0, 0, model.clock.time, U, κ)

    c_errors = compute_error(c_simulation, c_analytical)
    v_errors = compute_error(v_simulation, c_analytical)

    #####
    ##### Test u-advection
    #####
    
    ygrid = RegularCartesianGrid(size=(1, Nx, 1), x=(0, 1), y=(0, 2π), z=(0, 1), topology=topo)

    model = IncompressibleModel(architecture = architecture,
                                        grid = ygrid,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = nothing,
                                     closure = ConstantIsotropicDiffusivity(ν=κ, κ=κ))

    set!(model, v = U, 
                u = (x, y, z) -> c(y, x, z, 0, U, κ))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, progress_frequency=stop_iteration)

    println("Running Gaussian advection diffusion test for u with Nx = $Nx and Δt = $Δt...")
    run!(simulation)

    # Calculate errors
    u_simulation = model.velocities.u
    u_simulation = interior(u_simulation)[1, :, 1]
    u_errors = compute_error(u_simulation, c_analytical)

    return (

            c = (simulation = c_simulation,
                 analytical = c_analytical,
                         L₁ = c_errors.L₁,
                         L∞ = c_errors.L∞),

            v = (simulation = v_simulation,
                 analytical = c_analytical, # same solution as c.
                         L₁ = v_errors.L₁,
                         L∞ = v_errors.L∞),

            u = (simulation = u_simulation,
                 analytical = c_analytical, # same solution as c.
                         L₁ = u_errors.L₁,
                         L∞ = u_errors.L∞),

            grid = grid

            )
end

end # module
