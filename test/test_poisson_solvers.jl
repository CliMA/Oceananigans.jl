using Statistics: mean

using FFTW
using GPUifyLoops

using Oceananigans.Operators

function ∇²_ppn!(::Val{Dev}, Nx, Ny, Nz, Δx, Δy, Δz, f, ∇²f) where Dev
    @setup Dev

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds ∇²f[i, j, k] = ∇²_ppn(f, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
            end
        end
    end

    @synchronize
end

function test_mixed_fft_commutativity(N)
    A = rand(N, N, N)
    Ã1 = FFTW.dct(FFTW.rfft(A, [1, 2]), 3)
    Ã2 = FFTW.rfft(FFTW.dct(A, 3), [1, 2])
    Ã1 ≈ Ã2
end

function test_mixed_ifft_commutativity(N)
    A = rand(N, N, N)

    Ã1 = FFTW.dct(FFTW.rfft(A, [1, 2]), 3)
    Ã2 = FFTW.rfft(FFTW.dct(A, 3), [1, 2])

    A11 = FFTW.irfft(FFTW.idct(Ã1, 3), N, [1, 2])
    A12 = FFTW.idct(FFTW.irfft(Ã1, N, [1, 2]), 3)
    A21 = FFTW.irfft(FFTW.idct(Ã2, 3), N, [1, 2])
    A22 = FFTW.idct(FFTW.irfft(Ã2, N, [1, 2]), 3)
    A ≈ A11 && A ≈ A12 && A ≈ A21 && A ≈ A22
end

function test_fftw_planner(mm, Nx, Ny, Nz, planner_flag)
    g = RegularCartesianGrid(mm, (Nx, Ny, Nz), (100, 100, 100))

    RHS = CellField(mm, g, Complex{eltype(g)})
    solver = PoissonSolver(g, RHS, FFTW.PATIENT)

    true  # Just making sure our PoissonSolver does not spit an error.
end

function test_3d_poisson_ppn_planned!_div_free(mm, Nx, Ny, Nz, planner_flag)
    g = RegularCartesianGrid(mm, (Nx, Ny, Nz), (100, 100, 100))

    RHS = CellField(mm, g, Complex{eltype(g)})
    RHS_orig = CellField(mm, g, Complex{eltype(g)})
    ϕ = CellField(mm, g, Complex{eltype(g)})
    ∇²ϕ = CellField(mm, g, Complex{eltype(g)})

    RHS.data .= rand(Nx, Ny, Nz)
    RHS.data .= RHS.data .- mean(RHS.data)

    RHS_orig.data .= copy(RHS.data)

    solver = PoissonSolver(g, RHS, planner_flag)

    solve_poisson_3d_ppn_planned!(solver, g, RHS, ϕ)
    ∇²_ppn!(Val(mm.arch), Nx, Ny, Nz, g.Δx, g.Δy, g.Δz, ϕ, ∇²ϕ)

    ∇²ϕ.data ≈ RHS_orig.data
end
