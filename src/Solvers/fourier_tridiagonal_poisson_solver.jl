struct FourierTridiagonalPoissonSolver{A, G, B, S}
                  architecture :: A
                          grid :: G
    batched_tridiagonal_solver :: B
                       storage :: S
end

function FourierTridiagonalPoissonSolver(arch, grid)
    Nx, Ny, Nz = size(grid)
    # ΔzC, ΔzF = grid.ΔzC, grid.ΔzF

    λx = poisson_eigenvalues(grid.Nx, grid.Lx, 1, topology(grid, 1)())
    λy = poisson_eigenvalues(grid.Ny, grid.Ly, 2, topology(grid, 2)())

    # Lower and upper diagonals are the same
    ld = [1/ΔzF(1, 1, k, grid) for k in 1:Nz-1]
    ud = ld

    # Diagonal (different for each i,j)
    @inline δ(i, j, k, grid, λx, λy) = - (1/ΔzF(i, j, k-1, grid) + 1/ΔzF(i, j, k, grid)) - ΔzC(i, j, k, grid) * (λx + λy)

    d = zeros(Nx, Ny, Nz)
    for i in 1:Nx, j in 1:Ny
        d[i, j, 1] = -1/ΔzF(i, j, 1, grid) - ΔzC(i, j, 1, grid) * (λx[i] + λy[j])
        d[i, j, 2:Nz-1] .= [δ(i, j, k, grid, λx[i], λy[j]) for k in 2:Nz-1]
        d[i, j, Nz] = -1/ΔzF(i, j, Nz-1, grid) - ΔzC(i, j, Nz, grid) * (λx[i] + λy[j])
    end

    rhs_storage = arch_array(arch, zeros(complex(eltype(grid)), size(grid)...))
    btsolver = BatchedTridiagonalSolver(arch, dl=ld, d=d, du=ud, f=rhs_storage, grid=grid)

    sol_storage = arch_array(arch, zeros(complex(eltype(grid)), size(grid)...))

    return FourierTridiagonalPoissonSolver(arch, grid, btsolver, sol_storage)
end

function solve_poisson_equation!(solver::FourierTridiagonalPoissonSolver)
    ϕ = solver.storage
    RHS = solver.batched_tridiagonal_solver.f

    FFTW.fft!(RHS, [1, 2])

    solve_batched_tridiagonal_system!(ϕ, solver.architecture, solver.batched_tridiagonal_solver)

    FFTW.ifft!(ϕ, [1, 2])
    ϕ .= real.(ϕ)
    ϕ .= ϕ .- mean(ϕ)
   
    return nothing
end
