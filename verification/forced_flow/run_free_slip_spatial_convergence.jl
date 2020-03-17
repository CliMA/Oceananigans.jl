using Oceananigans

include("ForcedFlow/ForcedFlow.jl")

# Run 4 simulations:

ForcedFlow.FreeSlip.setup_and_run(Nx=16, Nz=16, CFL=1e-3)
ForcedFlow.FreeSlip.setup_and_run(Nx=32, Nz=32, CFL=1e-3)
ForcedFlow.FreeSlip.setup_and_run(Nx=64, Nz=64, CFL=1e-3)
ForcedFlow.FreeSlip.setup_and_run(Nx=128, Nz=128, CFL=1e-3)
