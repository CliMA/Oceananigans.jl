using Oceananigans

include("ConvergenceTests/ConvergenceTests.jl")

setup_and_run = ConvergenceTests.DoublyPeriodicFreeDecay.setup_and_run_xy

# Run 4 simulations:
Nx = [32, 64, 128, 256, 512]
stop_time = 0.01

# Calculate time step
 h = 2π / maximum(Nx)
Δt = 0.01 * h^2 # satisfy diffusive constraint
Nt = round(Int, stop_time / Δt)
Δt = stop_time / Nt

for N in Nx
    setup_and_run(Nx=N, Δt=Δt, stop_iteration=Nt)
end
