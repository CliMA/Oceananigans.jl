using Oceananigans

include("ConvergenceTests/ConvergenceTests.jl")

run_xy = ConvergenceTests.ForcedFlowFreeSlip.setup_and_run_xy
run_xz = ConvergenceTests.ForcedFlowFreeSlip.setup_and_run_xz

# Run 4 simulations:
Nx = [32, 64, 128, 256]
stop_time = 0.01

h = π / maximum(Nx)
Δt = 0.01 * h^2
Nt = round(Int, stop_time / Δt)
Δt = stop_time / Nt

for N in Nx
    run_xy(Nx=N, Δt=Δt, stop_iteration=stop_iteration)
    run_xz(Nx=N, Δt=Δt, stop_iteration=stop_iteration)
end
