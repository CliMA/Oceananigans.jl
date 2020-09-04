using Oceananigans

include("ConvergenceTests/ConvergenceTests.jl")

using .ConvergenceTests.ForcedFlowFreeSlip: setup_and_run_xy, setup_and_run_xz

# Run 4 simulations:
Nx = [32, 64, 128, 256]
stop_time = 0.01

h = π / maximum(Nx)
Δt = 0.01 * h^2
stop_iteration = round(Int, stop_time / Δt)
Δt = stop_time / stop_iteration

for N in Nx
    setup_and_run_xy(Nx=N, Δt=Δt, stop_iteration=stop_iteration)
    setup_and_run_xz(Nx=N, Δt=Δt, stop_iteration=stop_iteration)
end
