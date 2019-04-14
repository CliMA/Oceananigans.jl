using Statistics: mean

using FFTW
import GPUifyLoops: @launch, @loop, @synchronize

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
    grid = RegularCartesianGrid(ft, (Nx, Ny, Nz), (100, 100, 100))
    solver = PoissonSolver(CPU(), grid)
    true  # Just making sure our PoissonSolver does not error/crash.
end

function poisson_ppn_planned_div_free_cpu(ft, Nx, Ny, Nz, planner_flag)
    grid = RegularCartesianGrid(ft, (Nx, Ny, Nz), (100, 100, 100))
    solver = PoissonSolver(CPU(), grid)

    RHS = CellField(Complex{ft}, CPU(), grid)
    RHS_orig = CellField(Complex{ft}, CPU(), grid)
    ϕ = CellField(Complex{ft}, CPU(), grid)
    ∇²ϕ = CellField(Complex{ft}, CPU(), grid)

    @views RHS.data[1:Nx, 1:Ny, 1:Nz] .= rand(Nx, Ny, Nz)
    @views RHS.data[1:Nx, 1:Ny, 1:Nz] .= RHS.data[1:Nx, 1:Ny, 1:Nz] .- mean(RHS.data[1:Nx, 1:Ny, 1:Nz])

    RHS_orig.data .= copy(RHS.data)

    @views solver.storage .= RHS.data[1:Nx, 1:Ny, 1:Nz]

    solve_poisson_3d_ppn_planned!(solver, grid)

    @views ϕ.data[1:Nx, 1:Ny, 1:Nz] .= solver.storage

    ∇²_ppn!(grid, ϕ, ∇²ϕ)

    ∇²ϕ.data ≈ RHS_orig.data
end
