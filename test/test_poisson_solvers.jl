using Statistics: mean

using FFTW
using GPUifyLoops

using Oceananigans.Operators

@inline incmod1(a, n) = ifelse(a==n, 1, a + 1)
@inline decmod1(a, n) = ifelse(a==1, n, a - 1)

function ∇²_ppn!(g::RegularCartesianGrid, f::CellField, ∇²f::CellField)
    for k in 2:(g.Nz-1), j in 1:g.Ny, i in 1:g.Nx
       ∇²f.data[i, j, k] = (f.data[incmod1(i, g.Nx), j, k] - 2*f.data[i, j, k] + f.data[decmod1(i, g.Nx), j, k]) / g.Δx^2 +
                           (f.data[i, incmod1(j, g.Ny), k] - 2*f.data[i, j, k] + f.data[i, decmod1(j, g.Ny), k]) / g.Δy^2 +
                           (f.data[i, j, k+1]              - 2*f.data[i, j, k] + f.data[i, j, k-1])              / g.Δz^2
    end
    for j in 1:g.Ny, i in 1:g.Nx
        ∇²f.data[i, j,   1] = (f.data[i, j, 2] - f.data[i, j, 1]) / g.Δz^2 +
                              (f.data[incmod1(i, g.Nx), j, 1] - 2*f.data[i, j, 1] + f.data[decmod1(i, g.Nx), j, 1]) / g.Δx^2 +
                              (f.data[i, incmod1(j, g.Ny), 1] - 2*f.data[i, j, 1] + f.data[i, decmod1(j, g.Ny), 1]) / g.Δy^2
        ∇²f.data[i, j, end] = (f.data[i, j, end-1] - f.data[i, j, end]) / g.Δz^2 +
                              (f.data[incmod1(i, g.Nx), j, end] - 2*f.data[i, j, end] + f.data[decmod1(i, g.Nx), j, end]) / g.Δx^2 +
                              (f.data[i, incmod1(j, g.Ny), end] - 2*f.data[i, j, end] + f.data[i, decmod1(j, g.Ny), end]) / g.Δy^2
    end
    nothing
end

function laplacian3d_ppn(f)
    Nx, Ny, Nz = size(f)
    ∇²f = zeros(Nx, Ny, Nz)
    for k in 2:(Nz-1), j in 1:Ny, i in 1:Nx
       ∇²f[i, j, k] = f[incmod1(i, Nx), j, k] + f[decmod1(i, Nx), j, k] + f[i, incmod1(j, Ny), k] + f[i, decmod1(j, Ny), k] + f[i, j, k+1] + f[i, j, k-1] - 6*f[i, j, k]
    end
    for j in 1:Ny, i in 1:Nx
        ∇²f[i, j,   1] = -(f[i, j,     1] - f[i, j,   2]) + f[incmod1(i, Nx), j,   1] + f[decmod1(i, Nx), j,   1] + f[i, incmod1(j, Ny),   1] + f[i, decmod1(j, Ny),   1] - 4*f[i, j,   1]
        ∇²f[i, j, end] =  (f[i, j, end-1] - f[i, j, end]) + f[incmod1(i, Nx), j, end] + f[decmod1(i, Nx), j, end] + f[i, incmod1(j, Ny), end] + f[i, decmod1(j, Ny), end] - 4*f[i, j, end]
    end
    ∇²f
end

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

function test_3d_poisson_solver_ppn_div_free(Nx, Ny, Nz)
    f = rand(Nx, Ny, Nz)
    f .= f .- mean(f)
    ϕ = solve_poisson_3d_ppn(f, Nx, Ny, Nz, 1, 1, 1)
    laplacian3d_ppn(ϕ) ≈ f
end

function test_3d_poisson_solver_ppn!_div_free(mm, Nx, Ny, Nz)
    g = RegularCartesianGrid(mm, (Nx, Ny, Nz), (100, 100, 100))

    RHS = CellField(mm, g, Complex{eltype(g)})
    RHS_orig = CellField(mm, g, Complex{eltype(g)})
    ϕ = CellField(mm, g, Complex{eltype(g)})
    ∇²ϕ = CellField(mm, g, Complex{eltype(g)})

    RHS.data .= rand(Nx, Ny, Nz)
    RHS.data .= RHS.data .- mean(RHS.data)

    RHS_orig.data .= copy(RHS.data)

    solve_poisson_3d_ppn!(g, RHS, ϕ)

    ∇²_ppn!(g, ϕ, ∇²ϕ)

    ∇²ϕ.data ≈ RHS_orig.data
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
    # ∇²_ppn!(g, ϕ, ∇²ϕ)

    ∇²ϕ.data ≈ RHS_orig.data
end

function test_3d_poisson_solver_ppn_all_equal(mm, Nx, Ny, Nz, planner_flag)
    g = RegularCartesianGrid(mm, (Nx, Ny, Nz), (100, 100, 100))

    RHS = CellField(mm, g, Complex{eltype(g)})
    RHS_orig = CellField(mm, g, Complex{eltype(g)})
    ϕ = CellField(mm, g, Complex{eltype(g)})
    ∇²ϕ = CellField(mm, g, Complex{eltype(g)})

    RHS.data .= rand(Nx, Ny, Nz)
    RHS.data .= RHS.data .- mean(RHS.data)

    RHS_orig.data .= copy(RHS.data)

    solver = PoissonSolver(g, RHS, planner_flag)
end
