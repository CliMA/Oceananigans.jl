using Oceananigans

include("ConvergenceTests/ConvergenceTests.jl")

run = ConvergenceTests.TwoDimensionalDiffusion.run_simulation

# Setup and run 4 simulations
Nx = [32, 64, 128, 256]
stop_time = 0.5

# Calculate time step based on diffusive time-step constraint for finest mesh
     min_Δx = 2π / maximum(Nx)
proposal_Δt = 0.01 * min_Δx^2 # proposal time-step

stop_iteration = round(Int, stop_time / proposal_Δt)
            Δt = stop_time / stop_iteration # ensure time-stepping to exact finish time.

# Run simulations
for N in Nx
    run(Nx=N, Δt=Δt, stop_iteration=stop_iteration)
end
