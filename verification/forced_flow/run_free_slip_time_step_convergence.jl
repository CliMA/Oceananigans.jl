using Oceananigans

include("ForcedFlow/ForcedFlow.jl")

# Run 4 simulations:

ForcedFlow.FreeSlip.setup_and_run(Nx=32, Nz=32, CFL=1e-4)
ForcedFlow.FreeSlip.setup_and_run(Nx=32, Nz=32, CFL=5e-4)
#ForcedFlow.FreeSlip.setup_and_run(Nx=32, Nz=32, CFL=1e-3)
ForcedFlow.FreeSlip.setup_and_run(Nx=32, Nz=32, CFL=2e-3)
