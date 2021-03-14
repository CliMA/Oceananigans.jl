import PencilFFTs

import Oceananigans.Solvers: poisson_eigenvalues, solve_poisson_equation!

struct DistributedFFTBasedPoissonSolver{P, F, L, λ, S}
              plan :: P
         full_grid :: F
           my_grid :: L
       eigenvalues :: λ
           storage :: S
end

function DistributedFFTBasedPoissonSolver(arch, full_grid, local_grid)
    topo = (TX, TY, TZ) = topology(full_grid)

    λx = poisson_eigenvalues(full_grid.Nx, full_grid.Lx, 1, TX())
    λy = poisson_eigenvalues(full_grid.Ny, full_grid.Ly, 2, TY())
    λz = poisson_eigenvalues(full_grid.Nz, full_grid.Lz, 3, TZ())

    I, J, K = arch.local_index
    λx = λx[(J-1)*local_grid.Ny+1:J*local_grid.Ny, :, :]

    eigenvalues = (; λx, λy, λz)

    transform = PencilFFTs.Transforms.FFT!()
    proc_dims = (arch.ranks[2], arch.ranks[3])
    plan = PencilFFTs.PencilFFTPlan(size(full_grid), transform, proc_dims, MPI.COMM_WORLD)
    storage = PencilFFTs.allocate_input(plan)

    return DistributedFFTBasedPoissonSolver(plan, full_grid, local_grid, eigenvalues, storage)
end

function solve_poisson_equation!(solver::DistributedFFTBasedPoissonSolver)
    λx, λy, λz = solver.eigenvalues

    # Apply forward transforms.
    solver.plan * solver.storage

    # Solve the discrete Poisson equation.
    RHS = ϕ = solver.storage[2]
    @. ϕ = - RHS / (λx + λy + λz)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        ϕ[1, 1, 1] = 0
    end

    # Apply backward transforms.
    solver.plan \ solver.storage

    return nothing
end
