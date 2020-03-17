using Oceananigans

include("ConvergenceTests/ConvergenceTests.jl")

# Run 4 simulations:

#ConvergenceTests.ForcedFlowFreeSlip.setup_and_run_xz(Nx=16,  Nz=16,  CFL=1e-3)
#ConvergenceTests.ForcedFlowFreeSlip.setup_and_run_xz(Nx=32,  Nz=32,  CFL=1e-3)
#ConvergenceTests.ForcedFlowFreeSlip.setup_and_run_xz(Nx=64,  Nz=64,  CFL=1e-3)
#ConvergenceTests.ForcedFlowFreeSlip.setup_and_run_xz(Nx=128, Nz=128, CFL=1e-3)

ConvergenceTests.ForcedFlowFreeSlip.setup_and_run_xy(Nx=16,  Ny=16,  CFL=2e-3)
ConvergenceTests.ForcedFlowFreeSlip.setup_and_run_xy(Nx=32,  Ny=32,  CFL=2e-3)
ConvergenceTests.ForcedFlowFreeSlip.setup_and_run_xy(Nx=64,  Ny=64,  CFL=2e-3)
ConvergenceTests.ForcedFlowFreeSlip.setup_and_run_xy(Nx=128, Ny=128, CFL=2e-3)
