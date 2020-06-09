using Oceananigans

include("ConvergenceTests/ConvergenceTests.jl")

run_xy = ConvergenceTests.ForcedFlowFixedSlip.setup_and_run_xy

# Run 4 simulations:
Nx = [16, 32, 64, 128]
stop_time = 0.01

h = π / maximum(Nx)
Δt = 0.001 * h^2
stop_iteration = round(Int, stop_time / Δt)
Δt = stop_time / stop_iteration

for N in Nx
    run_xy(Nx=N, Δt=Δt, stop_iteration=stop_iteration)
end

include("analyze_forced_fixed_slip.jl")
