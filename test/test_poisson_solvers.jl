using Statistics: mean

using FFTW
using GPUifyLoops

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
