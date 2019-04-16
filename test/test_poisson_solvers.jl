using Statistics: mean

using FFTW
using OffsetArrays
import GPUifyLoops: @launch, @loop, @synchronize

@hascuda using CuArrays

using Oceananigans
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

function poisson_solver_initializes(arch, ft, Nx, Ny, Nz, planner_flag)
    grid = RegularCartesianGrid(ft, (Nx, Ny, Nz), (100, 100, 100))
    solver = PoissonSolver(arch, grid)
    true  # Just making sure our PoissonSolver does not error/crash.
end

function poisson_ppn_planned_div_free_cpu(ft, Nx, Ny, Nz, planner_flag)
    grid = RegularCartesianGrid(ft, (Nx, Ny, Nz), (100, 100, 100))
    solver = PoissonSolver(CPU(), grid)

    RHS = rand(Nx, Ny, Nz)
    RHS .= RHS .- mean(RHS)

    RHS_orig = copy(RHS)

    solver.storage .= RHS

    solve_poisson_3d_ppn_planned!(solver, grid)

    ϕ   = OffsetArray(zeros(Nx+2, Ny+2, Nz), 0:Nx+1, 0:Ny+1, 1:Nz)
    ∇²ϕ = OffsetArray(zeros(Nx+2, Ny+2, Nz), 0:Nx+1, 0:Ny+1, 1:Nz)

    @. @views ϕ[1:Nx, 1:Ny, 1:Nz] = real(solver.storage)

    Oceananigans.fill_halo_regions!(CPU(), grid, ϕ)

    ∇²_ppn!(grid, ϕ, ∇²ϕ)

    @views ∇²ϕ[1:Nx, 1:Ny, 1:Nz] ≈ RHS_orig
end

@hascuda begin
    function poisson_ppn_planned_div_free_gpu(ft, Nx, Ny, Nz)
        grid = RegularCartesianGrid(ft, (Nx, Ny, Nz), (100, 100, 100))
        solver = PoissonSolver(GPU(), grid)

        RHS = rand(Nx, Ny, Nz)
        RHS .= RHS .- mean(RHS)

        RHS_orig = copy(RHS)

        RHS = CuArray(RHS)
        RHS_orig = CuArray(RHS_orig)
        ϕ = CuArray(zeros(Nx, Ny, Nz))
        ∇²ϕ = CuArray(zeros(Nx, Ny, Nz))

        solver.storage .= RHS

        Tx, Ty = 16, 16  # Threads per block
        Bx, By, Bz = floor(Int, Nx/Tx), floor(Int, Ny/Ty), Nz  # Blocks in grid

        solve_poisson_3d_ppn_planned!(Tx, Ty, Bx, By, Bz, solver, grid)

        ϕ .= solver.storage

        @launch device(GPU()) ∇²_ppn!(grid, ϕ, ∇²ϕ, threads=(Tx, Ty), blocks=(Bx, By, Bz))

        ∇²ϕ ≈ RHS_orig
    end
end
