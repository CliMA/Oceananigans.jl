using PencilFFTs

struct DistributedFFTBasedPoissonSolver{P, F, L, λ, S}
              plan :: P
         full_grid :: F
           my_grid :: L
    my_eigenvalues :: λ
           storage :: S
end

reshaped_size(N, dim) = dim == 1 ? (N, 1, 1) :
                        dim == 2 ? (1, N, 1) :
                        dim == 3 ? (1, 1, N) : nothing

function poisson_eigenvalues(N, L, dim, ::Periodic)
    inds = reshape(1:N, reshaped_size(N, dim)...)
    return @. (2sin((inds - 1) * π / N) / (L / N))^2
end

function DistributedFFTBasedPoissonSolver(arch, full_grid, local_grid)
    topo = (TX, TY, TZ) = topology(full_grid)

    λx = poisson_eigenvalues(full_grid.Nx, full_grid.Lx, 1, TX())
    λy = poisson_eigenvalues(full_grid.Ny, full_grid.Ly, 2, TY())
    λz = poisson_eigenvalues(full_grid.Nz, full_grid.Lz, 3, TZ())

    I, J, K = arch.my_index
    my_eigenvalues = (
        λx = λx[(I-1)*local_grid.Nx+1:I*local_grid.Nx, :, :],
        λy = λy[:, (J-1)*local_grid.Ny+1:J*local_grid.Ny, :],
        λz = λz[:, :, (K-1)*local_grid.Nz+1:K*local_grid.Nz]
    )

    transform = PencilFFTs.Transforms.FFT!()
    proc_dims = (arch.ranks[2], arch.ranks[3])
    plan = PencilFFTPlan(size(full_grid), transform, proc_dims, MPI.COMM_WORLD)

    storage = allocate_input(plan)

    return DistributedFFTBasedPoissonSolver(plan, full_grid, local_grid, my_eigenvalues, storage)
end

function solve_poisson_equation!(solver::DistributedFFTBasedPoissonSolver)
    λx, λy, λz = solver.my_eigenvalues

    # https://jipolanco.github.io/PencilFFTs.jl/dev/PencilFFTs/#PencilFFTs.allocate_input
    RHS = ϕ = first(solver.storage)

    # Apply forward transforms.
    solver.plan * solver.storage

    # Solve the discrete Poisson equation.
    @. ϕ = -RHS / (λx + λy + λz)

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
