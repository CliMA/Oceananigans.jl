import PencilFFTs

using Oceananigans.Solvers: copy_real_component!
using Oceananigans.Distributed: rank2index

import Oceananigans.Solvers: poisson_eigenvalues, solve!
import Oceananigans.Architectures: architecture

struct DistributedFFTBasedPoissonSolver{P, F, L, λ, S}
              plan :: P
       global_grid :: F
        local_grid :: L
       eigenvalues :: λ
           storage :: S
end

architecture(solver::DistributedFFTBasedPoissonSolver) =
    architecture(solver.global_grid)

function DistributedFFTBasedPoissonSolver(global_grid, local_grid)

    arch = architecture(local_grid)
    arch.ranks[1] == arch.ranks[3] == 1 || @warn "Must have Rx == Rz == 1 for distributed fft solver"

    # Plan the PencilFFT
    communicator = MPI.COMM_WORLD
    pencil = PencilFFTs.Pencil(size(global_grid), communicator) # by default, decomposes along dims (2, 3).

    # Only works for fully-periodic:
    transform = PencilFFTs.Transforms.FFT!()

    plan = PencilFFTs.PencilFFTPlan(pencil, transform)

    # Allocate memory for in-place FFT + transpositions
    storage = PencilFFTs.allocate_input(plan)

    # Build _global_ eigenvalues
    topo = (TX, TY, TZ) = topology(global_grid)
    λx = poisson_eigenvalues(global_grid.Nx, global_grid.Lx, 1, TX())
    λy = poisson_eigenvalues(global_grid.Ny, global_grid.Ly, 2, TY())
    λz = poisson_eigenvalues(global_grid.Nz, global_grid.Lz, 3, TZ())

    # We add singleton dimensions because its "convenient", but
    # PencilFFTs doesn't want that.
    λx = dropdims(λx, dims=(2, 3))
    λy = dropdims(λy, dims=(1, 3))
    λz = dropdims(λz, dims=(1, 2))

    eigenvalues = PencilFFTs.localgrid(last(storage), (λx, λy, λz))

    return DistributedFFTBasedPoissonSolver(plan, global_grid, local_grid, eigenvalues, storage)
end

function solve!(x, solver::DistributedFFTBasedPoissonSolver)
    arch = architecture(solver)

    λx = solver.eigenvalues[1]
    λy = solver.eigenvalues[2]
    λz = solver.eigenvalues[3]

    # Apply forward transforms.
    solver.plan * solver.storage

    # Solve the discrete Poisson equation, storing the solution
    # temporarily in xc and later extracting the real part into 
    # the solution, x.
    xc = b = last(solver.storage)
    @. xc = - b / (λx + λy + λz)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        xc[1, 1, 1] = 0
    end

    # Apply backward transforms.
    solver.plan \ solver.storage
    xc_transposed = first(solver.storage)
	
    copy_event = launch!(arch, solver.local_grid, :xyz, copy_real_component!, x, xc_transposed, dependencies=device_event(arch))
    wait(device(arch), copy_event)

    return x
end

