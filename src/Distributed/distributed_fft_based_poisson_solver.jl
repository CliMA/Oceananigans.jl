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
    topo = (TX, TY, TZ) = topology(global_grid)

    λx = poisson_eigenvalues(global_grid.Nx, global_grid.Lx, 1, TX())
    λy = poisson_eigenvalues(global_grid.Ny, global_grid.Ly, 2, TY())
    λz = poisson_eigenvalues(global_grid.Nz, global_grid.Lz, 3, TZ())

    arch.ranks[1] == arch.ranks[3] == 1 || @warn "Must have Rx == Rz == 1 for distributed fft solver"

    Rx, Ry, Rz = arch.ranks

    # PencilFFT performs a permutation y -> x. 
    # x will be the "distributed direction" when  s = b / (λx + λy + λz)
    # we have to permute (Rx, Ry, Rz) with (Ry, Rx, Rz)
    I, J, K = rank2index(arch.local_rank, Ry, Rx, Rz)

    perm_Nx = global_grid.Nx ÷ Ry

    i₁ = (I-1) * perm_Nx + 1
    i₂ = I * perm_Nx
    λx = λx[i₁:i₂, :, :]

    eigenvalues = (; λx, λy, λz)

    transform = PencilFFTs.Transforms.FFT!()
    proc_dims = (arch.ranks[2], arch.ranks[3])
    plan = PencilFFTs.PencilFFTPlan(size(global_grid), transform, proc_dims, MPI.COMM_WORLD)
    storage = PencilFFTs.allocate_input(plan)

    return DistributedFFTBasedPoissonSolver(plan, global_grid, local_grid, eigenvalues, storage)
end

function solve!(x, solver::DistributedFFTBasedPoissonSolver)
    arch = architecture(solver)
    λx, λy, λz = solver.eigenvalues

    # Apply forward transforms.
    solver.plan * solver.storage

    # Solve the discrete Poisson equation, storing the solution
    # temporarily in xc and later extracting the real part into 
    # the solution, x.
    xc = b = solver.storage[2]
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
