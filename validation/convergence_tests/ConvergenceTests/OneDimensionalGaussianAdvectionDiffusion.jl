module OneDimensionalGaussianAdvectionDiffusion

using Printf
using Statistics
using LinearAlgebra

using Oceananigans
using Oceananigans.Grids

include("analysis.jl")

# Advection and diffusion of a Gaussian.
σ(t, κ, t₀) = 4 * κ * (t + t₀)
c(x, y, z, t, U, κ, t₀) = exp(-(x - U * t)^2 / 0.1^2)

function run_test(;
                  Nx,
                  Δt,
                  stop_iteration,
                  U = 1,
                  κ = 1e-4,
                  width = 0.05,
                  architecture = CPU(),
                  topo = (Periodic, Periodic, Periodic),
                  advection = CenteredSecondOrder()
                  )

    t₀ = width^2 / 4κ

    domain = (x=(-1, 1.5), y=(0, 1), z=(0, 1))
    grid = RegularCartesianGrid(topology=topo, size=(Nx, 1, 1), halo=(3, 3, 3); domain...)

    model = IncompressibleModel(architecture = architecture,
#                                 timestepper = :RungeKutta3,
                                 timestepper = :QuasiAdamsBashforth2,
                                        grid = grid,
                                   advection = advection,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = :c,
                                     closure = nothing)

    set!(model, u = U,
                c = (x, y, z) -> c(x, y, z, 0, U, κ, t₀),
         )

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running Gaussian advection test for cx with Nx = $Nx and Δt = $Δt ($(typeof(advection)))..."
    run!(simulation)

    x = xnodes(model.tracers.c)
    c_analytical = c.(x, 0, 0, Δt, U, κ, t₀)

    # Calculate errors
    cx_simulation = interior(model.tracers.c)[:, 1, 1]
    cx_error_L₁   = norm(cx_simulation .- c_analytical, 1  )/Nx
    cx_error_L∞   = norm(cx_simulation .- c_analytical, Inf)/Nx^(1/Inf)
    
    #print("cx_error_L₁  = ", cx_error_L₁, "\n")
    #print("cx_error_L∞  = ", cx_error_L∞, "\n")
    
    return (

            cx = (simulation = cx_simulation,
                  analytical = c_analytical,
                  L₁ = cx_error_L₁,
                  L∞ = cx_error_L∞
                  ),
        
            grid = grid

            )
end

end # module
