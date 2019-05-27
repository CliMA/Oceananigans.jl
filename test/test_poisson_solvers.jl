using FFTW
using Statistics: mean
using LinearAlgebra: norm

import GPUifyLoops: @launch, @loop, @unroll, @synchronize

using Oceananigans.Operators

function ∇²_ppn!(grid::RegularCartesianGrid, f, ∇²f)
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds ∇²f[i, j, k] = ∇²_ppn(grid, f, i, j, k)
            end
        end
    end

    @synchronize
end

function mixed_fft_commutes(N)
    A = rand(N, N, N)
    Ã1 = FFTW.dct(FFTW.rfft(A, [1, 2]), 3)
    Ã2 = FFTW.rfft(FFTW.dct(A, 3), [1, 2])
    Ã1 ≈ Ã2
end

function mixed_ifft_commutes(N)
    A = rand(N, N, N)

    Ã1 = FFTW.dct(FFTW.rfft(A, [1, 2]), 3)
    Ã2 = FFTW.rfft(FFTW.dct(A, 3), [1, 2])

    A11 = FFTW.irfft(FFTW.idct(Ã1, 3), N, [1, 2])
    A12 = FFTW.idct(FFTW.irfft(Ã1, N, [1, 2]), 3)
    A21 = FFTW.irfft(FFTW.idct(Ã2, 3), N, [1, 2])
    A22 = FFTW.idct(FFTW.irfft(Ã2, N, [1, 2]), 3)
    A ≈ A11 && A ≈ A12 && A ≈ A21 && A ≈ A22
end

function fftw_planner_works(ft, Nx, Ny, Nz, planner_flag)
    g = RegularCartesianGrid(ft, (Nx, Ny, Nz), (100, 100, 100))
    RHS = CellField(Complex{ft}, CPU(), g)
    solver = PoissonSolver(g, RHS, FFTW.ESTIMATE)
    true  # Just making sure our PoissonSolver does not error/crash.
end

function poisson_ppn_planned_div_free_cpu(ft, Nx, Ny, Nz, planner_flag)
    g = RegularCartesianGrid(ft, (Nx, Ny, Nz), (100, 100, 100))

    RHS = CellField(Complex{ft}, CPU(), g)
    RHS_orig = CellField(Complex{ft}, CPU(), g)
    ϕ = CellField(Complex{ft}, CPU(), g)
    ∇²ϕ = CellField(Complex{ft}, CPU(), g)

    RHS.data .= rand(Nx, Ny, Nz)
    RHS.data .= RHS.data .- mean(RHS.data)

    RHS_orig.data .= copy(RHS.data)

    solver = PoissonSolver(g, RHS, planner_flag)

    solve_poisson_3d_ppn_planned!(solver, g, RHS, ϕ)
    ∇²_ppn!(g, ϕ, ∇²ϕ)

    ∇²ϕ.data ≈ RHS_orig.data
end

function poisson_ppn_planned_div_free_gpu(ft, Nx, Ny, Nz)
    grid = RegularCartesianGrid(ft, (Nx, Ny, Nz), (100, 100, 100))

    RHS = CellField(Complex{ft}, GPU(), grid)
    RHS_orig = CellField(Complex{ft}, GPU(), grid)
    ϕ = CellField(Complex{ft}, GPU(), grid)
    ∇²ϕ = CellField(Complex{ft}, GPU(), grid)

    solver = init_poisson_solver(GPU(), grid, RHS)

    RHS.data .= CuArray(rand(Nx, Ny, Nz))
    RHS.data .= RHS.data .- mean(RHS.data)

    RHS_orig.data .= copy(RHS.data)

    # Performing the permutation [a, b, c, d, e, f] -> [a, c, e, f, d, b] in the z-direction in preparation to calculate
    # the DCT in the Poisson solver.
    RHS.data .= cat(RHS.data[:, :, 1:2:Nz], RHS.data[:, :, Nz:-2:2]; dims=3)

    Tx, Ty = 16, 16
    Bx, By, Bz = floor(Int, Nx/Tx), floor(Int, Ny/Ty), Nz  # Blocks in grid
    solve_poisson_3d_ppn_gpu_planned!(Tx, Ty, Bx, By, Bz, solver, grid, RHS, ϕ)

    # Undoing the permutation made above to complete the IDCT.
    ϕ.data .= CuArray{eltype(ϕ.data)}(reshape(permutedims(cat(ϕ.data[:, :, 1:Int(Nz/2)], f[:, :, end:-1:Int(Nz/2)+1]; dims=4), (1, 2, 4, 3)), Nx, Ny, Nz))
    @. ϕ.data = real(ϕ.data)

    ∇²_ppn!(grid, ϕ, ∇²ϕ)

    ∇²ϕ.data ≈ RHS_orig.data
end

"""
    Test that the Poisson solver can recover an analytic solution. In this test, we are trying to see if the solver
    can recover the solution ``\\Psi(x, y, z) = cos(\\pi m_z z / L_z) sin(2\\pi m_y y / L_y) sin(2\\pi m_x x / L_x)``
    by giving it the source term or right hand side (RHS), which is ``f(x, y, z) = \\nabla^2 \\Psi(x, y, z) =
    -((\\pi m_z / L_z)^2 + (2\\pi m_y / L_y)^2 + (2\\pi m_x/L_x)^2) \\Psi(x, y, z)``.
"""
function poisson_ppn_recover_sine_cosine_solution(ft, Nx, Ny, Nz, Lx, Ly, Lz, mx, my, mz)
    grid = RegularCartesianGrid(ft, (Nx, Ny, Nz), (Lx, Ly, Lz))

    RHS = CellField(Complex{ft}, CPU(), grid)
    ϕ = CellField(Complex{ft}, CPU(), grid)

    solver = PoissonSolver(grid, RHS)

    xC, yC, zC = grid.xC, grid.yC, grid.zC
    xC = reshape(xC, (Nx, 1, 1))
    yC = reshape(yC, (1, Ny, 1))
    zC = reshape(zC, (1, 1, Nz))

    Ψ(x, y, z) = cos(π*mz*z/Lz) * sin(2π*my*y/Ly) * sin(2π*mx*x/Lx)
    f(x, y, z) = -((mz*π/Lz)^2 + (2π*my/Ly)^2 + (2π*mx/Lx)^2) * Ψ(x, y, z)

    @. RHS.data = f(xC, yC, zC)
    solve_poisson_3d_ppn_planned!(solver, grid, RHS, ϕ)

    error = norm(ϕ.data - Ψ.(xC, yC, zC)) / √(Nx*Ny*Nz)

    @info "Error (ℓ²-norm), $ft, N=($Nx, $Ny, $Nz), m=($mx, $my, $mz): $error"

    isapprox(real.(ϕ.data),  Ψ.(xC, yC, zC); rtol=5e-2)
end
