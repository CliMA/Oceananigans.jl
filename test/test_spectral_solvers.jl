using Statistics: mean

using FFTW

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
    ssp = SpectralSolverParameters(g, RHS, FFTW.PATIENT)

    true  # Just making sure our SpectralSolverParameters does not spit an error.
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

    ssp = SpectralSolverParameters(g, RHS, planner_flag)

    solve_poisson_3d_ppn_planned!(ssp, g, RHS, ϕ)
    ∇²_ppn!(g, ϕ, ∇²ϕ)

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

    ssp = SpectralSolverParameters(g, RHS, planner_flag)
end
