struct FourierTridiagonalPoissonSolver{A, G, B, S}
                  architecture :: A
                          grid :: G
    batched_tridiagonal_solver :: B
                       storage :: S
end

@kernel function compute_diagonals!(D, grid, λx, λy)
    i, j = @index(Global, NTuple)
    Nz = grid.Nz

    D[i, j, 1] = -1/ΔzF(i, j, 1, grid) - ΔzC(i, j, 1, grid) * (λx[i] + λy[j])

    @unroll for k in 2:Nz-1
        D[i, j, k] = - (1/ΔzF(i, j, k-1, grid) + 1/ΔzF(i, j, k, grid)) - ΔzC(i, j, k, grid) * (λx[i] + λy[j])
    end

    D[i, j, Nz] = -1/ΔzF(i, j, Nz-1, grid) - ΔzC(i, j, Nz, grid) * (λx[i] + λy[j])
end

function FourierTridiagonalPoissonSolver(arch, grid)
    # FIXME: Gonna add more tests before supporting all/other topologies.
    @assert topology(grid) == (Periodic, Periodic, Bounded)

    Nx, Ny, Nz = size(grid)

    # Compute discrete Poisson eigenvalues
    λx = poisson_eigenvalues(grid.Nx, grid.Lx, 1, topology(grid, 1)())
    λy = poisson_eigenvalues(grid.Ny, grid.Ly, 2, topology(grid, 2)())

    λx = arch_array(arch, λx)
    λy = arch_array(arch, λy)

    # Lower and upper diagonals are the same
    CUDA.@allowscalar begin
        ld = arch_array(arch, [1/ΔzF(1, 1, k, grid) for k in 1:Nz-1])
        ud = ld
    end

    # Compute diagonal coefficients for each grid point
    D = arch_array(arch, zeros(Nx, Ny, Nz))
    event = launch!(arch, grid, :xy, compute_diagonals!, D, grid, λx, λy,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    # Set up batched tridiagonal solver
    rhs_storage = arch_array(arch, zeros(complex(eltype(grid)), size(grid)...))
    btsolver = BatchedTridiagonalSolver(arch, dl=ld, d=D, du=ud, f=rhs_storage, grid=grid)

    sol_storage = arch_array(arch, zeros(complex(eltype(grid)), size(grid)...))

    return FourierTridiagonalPoissonSolver(arch, grid, btsolver, sol_storage)
end

function solve_poisson_equation!(solver::FourierTridiagonalPoissonSolver)
    ϕ = solver.storage
    RHS = solver.batched_tridiagonal_solver.f

    solver.architecture isa CPU && FFTW.fft!(RHS, [1, 2])
    solver.architecture isa GPU && CUDA.CUFFT.fft!(RHS, [1, 2])

    solve_batched_tridiagonal_system!(ϕ, solver.architecture, solver.batched_tridiagonal_solver)

    solver.architecture isa CPU && FFTW.ifft!(ϕ, [1, 2])
    solver.architecture isa GPU && CUDA.CUFFT.ifft!(ϕ, [1, 2])

    ϕ .= real.(ϕ)
    ϕ .= ϕ .- mean(ϕ)

    return nothing
end
