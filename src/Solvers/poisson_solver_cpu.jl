"""
    plan_transforms(::PPN, A::Array; planner_flag=FFTW.PATIENT)

Return the transforms required to solve Poisson's equation with periodic
boundary conditions in the x,y dimensions and staggered Neumann boundary
conditions in the z-dimension.

Fast Fourier transforms (FFTs) are used in the x,y dimensions and real-to-real
discrete cosine transforms are used in the z dimension. Note that the DCT-II is
used for the DCT and the DCT-III for the IDCT which correspond to REDFT10 and
REDFT01 in FFTW.

They operatore on an array with the shape of `A`, which is needed to plan
efficient transforms. `A` will be mutated.

Note that the transforms returns operate on Arrays and so only work on CPUs.
"""
function plan_transforms(::PPN, A::Array, planner_flag=FFTW.PATIENT)
    FFT!  = FFTW.plan_fft!(A, [1, 2]; flags=planner_flag)
    IFFT! = FFTW.plan_ifft!(A, [1, 2]; flags=planner_flag)
    DCT!  = FFTW.plan_r2r!(A, FFTW.REDFT10, 3; flags=planner_flag)
    IDCT! = FFTW.plan_r2r!(A, FFTW.REDFT01, 3; flags=planner_flag)
    return FFT!, IFFT!, DCT!, IDCT!
end

"""
    plan_transforms(::PPN, A::Array; planer_flag=FFTW.PATIENT)

Similar to `plan_transforms(::PPN, ...)` but return the transforms required to
solve Poisson's equation with periodic boundary conditions in the x dimension
and staggered Neumann boundary conditions in the y,z dimensions.
"""
function plan_transforms(::PNN, A::Array, planner_flag=FFTW.PATIENT)
    FFT!  = FFTW.plan_fft!(A, 1; flags=planner_flag)
    IFFT! = FFTW.plan_ifft!(A, 1; flags=planner_flag)
    DCT!  = FFTW.plan_r2r!(A, FFTW.REDFT10, [2, 3]; flags=planner_flag)
    IDCT! = FFTW.plan_r2r!(A, FFTW.REDFT01, [2, 3]; flags=planner_flag)
    return FFT!, IFFT!, DCT!, IDCT!
end

struct PoissonSolverCPU{BC, AAR, AAC, FFTT, DCTT, IFFTT, IDCTT} <: AbstractPoissonSolver
        bcs :: BC
        kx² :: AAR
        ky² :: AAR
        kz² :: AAR
    storage :: AAC
       FFT! :: FFTT
       DCT! :: DCTT
      IFFT! :: IFFTT
      IDCT! :: IDCTT
end

function PoissonSolverCPU(pbcs::PoissonBCs, grid::AbstractGrid, planner_flag=FFTW.PATIENT)
    Nx, Ny, Nz, _ = unpack_grid(grid)

    # The eigenvalues of the discrete form of Poisson's equation correspond
    # to discrete wavenumbers so we call them k² to make the analogy with
    # solving Poisson's equation in spectral space.
    kx² = λi(grid, pbcs)
    ky² = λj(grid, pbcs)
    kz² = λk(grid, pbcs)

    # Storage for RHS and Fourier coefficients is hard-coded to be Float64
    # because of precision issues with Float32.
    # See https://github.com/climate-machine/Oceananigans.jl/issues/55
    storage = zeros(Complex{Float64}, grid.Nx, grid.Ny, grid.Nz)

    FFT!, IFFT!, DCT!, IDCT! = plan_transforms(pbcs, storage, planner_flag)

    PoissonSolverCPU(pbcs, kx², ky², kz², storage, FFT!, DCT!, IFFT!, IDCT!)
end

normalize_idct_output(::PPN, grid::AbstractGrid, ϕ::Array) = (@. ϕ = ϕ / (2grid.Nz))
normalize_idct_output(::PNN, grid::AbstractGrid, ϕ::Array) = (@. ϕ = ϕ / (4grid.Ny*grid.Nz))

"""
    solve_poisson_3d!(solver::PoissonSolverCPU, grid::RegularCartesianGrid)

Solve Poisson equation on a uniform staggered grid (Arakawa C-grid) with
appropriate boundary conditions as specified by `solver.bcs` using planned FFTs
and DCTs. The right-hand-side RHS is stored in solver.storage which the solver
mutates to produce the solution, so it will also be stored in solver.storage.

We should describe the algorithm in detail in the documentation.
"""
function solve_poisson_3d!(solver::PoissonSolverCPU, grid::RegularCartesianGrid)
    Nx, Ny, Nz, _ = unpack_grid(grid)
    kx², ky², kz² = solver.kx², solver.ky², solver.kz²

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    solver.DCT! * RHS  # Calculate DCTᶻ(f) in place.
    solver.FFT! * RHS  # Calculate FFTˣʸ(f) in place.

    # Solve the discrete Poisson equation in spectral space. We are essentially
    # computing the Fourier coefficients of the solution from the Fourier
    # coefficients of the RHS.
    @. ϕ = -RHS / (kx² + ky² + kz²)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    ϕ[1, 1, 1] = 0

    solver.IFFT! * ϕ  # Calculate IFFTˣʸ(ϕ) in place.
    solver.IDCT! * ϕ  # Calculate IDCTᶻ(ϕ) in place.

    # Must normalize by 2N for each dimension transformed via FFTW.REDFT.
    normalize_idct_output(solver.bcs, grid, ϕ)

    nothing
end
