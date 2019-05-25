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

Oceananigans.PoissonSolver(::CPU, grid, ϕ, args...; kwargs...) = PoissonSolver(grid, ϕ, args...; kwargs...)
Oceananigans.PoissonSolver(::GPU, grid, ϕ, args...; kwargs...) = PoissonSolverGPU(grid, ϕ; kwargs...)

function fftw_planner_works(ft, Nx, Ny, Nz, planner_flag, arch=CPU())
    g = RegularCartesianGrid(ft, (Nx, Ny, Nz), (100, 100, 100))
    RHS = CellField(Complex{ft}, arch, g)
    solver = PoissonSolver(arch, g, RHS, FFTW.ESTIMATE)
    true  # Just making sure our PoissonSolver does not error/crash.
end

function solve_poisson!(ϕ, solver::PoissonSolver, grid, rhs, ϕcomplex, args...) 
    solve_poisson_3d_ppn_planned!(solver, grid, rhs, ϕcomplex)
    @. ϕ.data = real(ϕcomplex.data)
    return nothing
end

function solve_poisson!(ϕ, solver::PoissonSolverGPU, grid, rhs, ϕcomplex, Tx, Ty, Bx, By, Bz) #threads, blocks)
    solve_poisson_3d_ppn_gpu_planned!(Tx, Ty, Bx, By, Bz, solver, grid, rhs, ϕcomplex)
    @launch device(GPU()) Oceananigans.idct_permute!(grid, ϕcomplex.data, ϕ.data, threads=(Tx, Ty), blocks=(Bx, By, Bz))
    return nothing
end

function poisson_ppn_planned_div_free(T, Nx, Ny, Nz, planner_flag, arch=CPU(); 
                                        Tx=16, Ty=16, Bx=floor(Int, Nx/Tx), By=floor(Int, Ny/Ty), Bz=Nz)

    grid = RegularCartesianGrid(T, (Nx, Ny, Nz), (7.1, 6.3, 5.6))

         rhs = CellField(Complex{T}, arch, grid)
         tmp = CellField(Complex{T}, arch, grid)
    rhs_orig = CellField(T, arch, grid)
           ϕ = CellField(T, arch, grid)
         ∇²ϕ = CellField(T, arch, grid)

    if arch == CPU()
        random_rhs = rand(T, Nx, Ny, Nz)
    else
        random_rhs = CuArray{T}(rand(Nx, Ny, Nz))
    end

    @. rhs.data .= random_rhs
    rhs.data .= rhs.data .- mean(rhs.data)
    @. rhs_orig.data .= real(rhs.data) # Store original array (because rhs data is destroyed... ?)

    solver = PoissonSolver(arch, grid, rhs, planner_flag)

    solve_poisson!(ϕ, solver, grid, rhs, tmp, Tx, Ty, Bx, By, Bz)
    @launch device(arch) ∇²_ppn!(grid, ϕ.data, ∇²ϕ.data, threads=(Tx, Ty), blocks=(Bx, By, Bz))

    error = norm(∇²ϕ.data - rhs_orig.data) / √(Nx*Ny*Nz)
    @info "Random poisson solve error (ℓ²-norm) $(arch), $T, N=($Nx, $Ny, $Nz): $error"

    ∇²ϕ.data ≈ rhs_orig.data
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
