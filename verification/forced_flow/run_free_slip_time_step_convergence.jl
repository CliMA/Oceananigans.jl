using Oceananigans

include("Convergence/Convergence.jl")

# Run 4 simulations:

ConvergenceForcedFlowFreeSlip.setup_and_run_xz(Nx=32, Nz=32, CFL=1e-4)
ConvergenceForcedFlowFreeSlip.setup_and_run_xz(Nx=32, Nz=32, CFL=5e-4)
ConvergenceForcedFlowFreeSlip.setup_and_run_xz(Nx=32, Nz=32, CFL=1e-3)
ConvergenceForcedFlowFreeSlip.setup_and_run_xz(Nx=32, Nz=32, CFL=2e-3)
