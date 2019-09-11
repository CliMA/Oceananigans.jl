import FFTW
import GPUifyLoops: @launch, @loop

abstract type PoissonBCs end

# PoissonBCs are named XYZ, where each of X, Y, and Z is either
# 'P' (for Periodic) or 'N' (for Neumann).
struct PPN <: PoissonBCs end
struct PNN <: PoissonBCs end

# Not yet supported:
#struct PPP <: PoissonBCs end
#struct NNN <: PoissonBCs end
#struct NPP <: PoissonBCs end
#struct PNP <: PoissonBCs end
#struct NPN <: PoissonBCs end
#struct NNP <: PoissonBCs end

poisson_bc_symbol(::BC) = :N
poisson_bc_symbol(::BC{<:Periodic}) = :P

"""
    PoissonBCs(bcs)

Returns the boundary conditions for the Poisson solver corresponding
to the model boundary conditions `bcs`.
"""
function PoissonBCs(bcs)
    # We assume that bounary conditions on all fields are
    # consistent with the boundary conditions on one side of ``u``.
    x = poisson_bc_symbol(bcs.u.x.left)
    y = poisson_bc_symbol(bcs.u.y.left)
    z = poisson_bc_symbol(bcs.u.z.left)

    return eval(Expr(:call, Symbol(x, y, z)))
end

PoissonSolver(::CPU, pbcs::PoissonBCs, grid::Grid) = PoissonSolverCPU(pbcs, grid)
PoissonSolver(::GPU, pbcs::PoissonBCs, grid::Grid) = PoissonSolverGPU(pbcs, grid)

unpack_grid(grid::Grid) = grid.Nx, grid.Ny, grid.Nz, grid.Lx, grid.Ly, grid.Lz

"""
    ω(M, k)

Return the `M`th root of unity raised to the `k`th power.
"""
@inline ω(M, k) = exp(-2im*π*k/M)

"""
    λi(grid::Grid, ::PoissonBCs)

Return an Nx×1×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with periodic boundary conditions in the x-dimension on `grid`.
"""
function λi(grid::Grid, ::PoissonBCs)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    is = reshape(1:Nx, Nx, 1, 1)
    @. (2sin((is-1)*π/Nx) / (Lx/Nx))^2
end

"""
    λj(grid::Grid, ::PPN)

Return an 1×Ny×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with periodic boundary conditions in the y-dimension on `grid`.
"""
function λj(grid::Grid, ::PPN)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    js = reshape(1:Ny, 1, Ny, 1)
    @. (2sin((js-1)*π/Ny) / (Ly/Ny))^2
end

"""
    λj(grid::Grid, ::PNN)

Return an 1×Ny×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with staggered Neumann boundary conditions in the y-dimension on `grid`.
"""
function λj(grid::Grid, ::PNN)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    js = reshape(1:Ny, 1, Ny, 1)
    @. (2sin((js-1)*π/(2Ny)) / (Ly/Ny))^2
end

"""
    λk(grid::Grid, ::PoissonBCs)

Return an 1×1×Nz array of eigenvalues satisfying the discrete form of Poisson's
equation with staggered Neumann boundary conditions in the y-dimension on `grid`.
"""
function λk(grid::Grid, ::PoissonBCs)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    ks = reshape(1:Nz, 1, 1, Nz)
    @. (2sin((ks-1)*π/(2Nz)) / (Lz/Nz))^2
end

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

struct PoissonSolverCPU{BC, AAR, AAC, FFTT, DCTT, IFFTT, IDCTT} <: PoissonSolver
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

function PoissonSolverCPU(pbcs::PoissonBCs, grid::Grid, planner_flag=FFTW.PATIENT)
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

normalize_idct_output(::PPN, grid::Grid, ϕ::Array) = (@. ϕ = ϕ / (2grid.Nz))
normalize_idct_output(::PNN, grid::Grid, ϕ::Array) = (@. ϕ = ϕ / (4grid.Ny*grid.Nz))

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

function plan_transforms(::PPN, A)
    FFT!      = plan_fft!(A, [1, 2])
    FFT_DCT!  = plan_fft!(A, 3)
    IFFT!     = plan_ifft!(A, [1, 2])
    IFFT_DCT! = plan_ifft!(A, 3)
    return FFT!, FFT_DCT!, IFFT!, IFFT_DCT!
end

function plan_transforms(::PNN, A)
    FFT!      = plan_fft!(A, 1)
    FFT_DCT!  = plan_fft!(A, [2, 3])
    IFFT!     = plan_ifft!(A, 1)
    IFFT_DCT! = plan_ifft!(A, [2, 3])
    return FFT!, FFT_DCT!, IFFT!, IFFT_DCT!
end

struct PoissonSolverGPU{BC, AAR, AAC, AAI, FFT, FFTD, IFFT, IFFTD} <: PoissonSolver
          bcs :: BC
          kx² :: AAR
          ky² :: AAR
          kz² :: AAR
       ω_4Ny⁺ :: AAC
       ω_4Ny⁻ :: AAC
       ω_4Nz⁺ :: AAC
       ω_4Nz⁻ :: AAC
     p_y_inds :: AAI
     p_z_inds :: AAI
     r_y_inds :: AAI
     r_z_inds :: AAI
         M_ky :: AAR
         M_kz :: AAR
      storage :: AAC
     storage2 :: AAC
         FFT! :: FFT
     FFT_DCT! :: FFTD
        IFFT! :: IFFT
    IFFT_DCT! :: IFFTD
end

function PoissonSolverGPU(pbcs::PoissonBCs, grid::Grid)
    Nx, Ny, Nz, _ = unpack_grid(grid)

    # The eigenvalues of the discrete form of Poisson's equation correspond
    # to discrete wavenumbers so we call them k² to make the analogy with
    # solving Poisson's equation in spectral space.
    kx² = λi(grid, pbcs) |> CuArray
    ky² = λj(grid, pbcs) |> CuArray
    kz² = λk(grid, pbcs) |> CuArray

    # Storage for RHS and Fourier coefficients is hard-coded to be Float64
    # because of precision issues with Float32.
    # See https://github.com/climate-machine/Oceananigans.jl/issues/55
    storage = zeros(Complex{Float64}, grid.Nx, grid.Ny, grid.Nz) |> CuArray

    # For solving the Poisson equation with PNN boundary conditions, we need a
    # second storage/buffer array to perform the 2D fast cosine transform. Maybe
    # we can get around this but for now we'll just use up a second array. It's
    # not needed for the PPN case so we just use the smallest 3D array we can
    # to keep the struct concretely typed.
    if pbcs == PPN()
        storage2 = zeros(Complex{Float64}, 1, 1, 1) |> CuArray
    elseif pbcs == PNN()
        storage2 = zeros(Complex{Float64}, grid.Nx, grid.Ny, grid.Nz) |> CuArray
    end

    ky⁺ = reshape(0:Ny-1,       1, Ny, 1)
    kz⁺ = reshape(0:Nz-1,       1, 1, Nz)
    ky⁻ = reshape(0:-1:-(Ny-1), 1, Ny, 1)
    kz⁻ = reshape(0:-1:-(Nz-1), 1, 1, Nz)

    ω_4Ny⁺ = ω.(4Ny, ky⁺) |> CuArray
    ω_4Nz⁺ = ω.(4Nz, kz⁺) |> CuArray
    ω_4Ny⁻ = ω.(4Ny, ky⁻) |> CuArray
    ω_4Nz⁻ = ω.(4Nz, kz⁻) |> CuArray

    # The zeroth coefficient of the IDCT (DCT-III or FFTW.REDFT01) is not
    # multiplied by 2. For some reason, we only need to account for this when
    # doing a 1D IDCT (for PPN boundary conditions) but not for a 2D IDCT. It's
    # possible that the masks are effectively doing this job.
    if pbcs == PPN()
        ω_4Nz⁻[1] *= 1/2
    end

    # Indices used when we need views to permuted arrays where the odd indices
    # are iterated over first followed by the even indices.
    p_y_inds = [1:2:Ny..., Ny:-2:2...] |> CuArray
    p_z_inds = [1:2:Nz..., Nz:-2:2...] |> CuArray

    # Indices used when we need views with reverse indexing but index N+1 should
    # return a 0. This can't be enforced using views so we just map N+1 to 1,
    # and use masks M_ky and M_kz to enforce that the value at N+1 is 0.
    r_y_inds = [1, collect(Ny:-1:2)...] |> CuArray
    r_z_inds = [1, collect(Nz:-1:2)...] |> CuArray

    M_ky = ones(1, Ny, 1) |> CuArray
    M_kz = ones(1, 1, Nz) |> CuArray

    M_ky[1] = 0
    M_kz[1] = 0

    FFT!, FFT_DCT!, IFFT!, IFFT_DCT! = plan_transforms(pbcs, storage)

    PoissonSolverGPU(pbcs, kx², ky², kz², ω_4Ny⁺, ω_4Ny⁻, ω_4Nz⁺, ω_4Nz⁻,
                     p_y_inds, p_z_inds, r_y_inds, r_z_inds, M_ky, M_kz,
                     storage, storage2, FFT!, FFT_DCT!, IFFT!, IFFT_DCT!)
end

"""
    solve_poisson_3d!(solver::PoissonSolverGPU, grid::RegularCartesianGrid)

Similar to solve_poisson_3d!(solver::PoissonSolverCPU, ...) except that since
the discrete cosine transform is not available through cuFFT, we perform our
own fast cosine transform (FCT) via an algorithm that utilizes the FFT.

Note that for the FCT algorithm to work, the input must have been permuted along
the dimension the FCT is to be calculated by ordering the odd elements first
followed by the even elements. For example,

    [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]

The output will be permuted in this way and so the permutation must be undone.

We should describe the algorithm in detail in the documentation.
"""
function solve_poisson_3d!(solver::PoissonSolverGPU{<:PPN}, grid::RegularCartesianGrid)
    kx², ky², kz² = solver.kx², solver.ky², solver.kz²
    ω_4Ny⁺, ω_4Ny⁻, ω_4Nz⁺, ω_4Nz⁻ = solver.ω_4Ny⁺, solver.ω_4Ny⁻, solver.ω_4Nz⁺, solver.ω_4Nz⁻

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    # Calculate DCTᶻ(f) in place using the FFT.
    solver.FFT_DCT! * RHS
    @. RHS = 2 * real(ω_4Nz⁺ * RHS)

    solver.FFT! * RHS  # Calculate FFTˣʸ(f) in place.

    # Solve the discrete Poisson equation in spectral space. We are essentially
    # computing the Fourier coefficients of the solution from the Fourier
    # coefficients of the RHS.
    @. ϕ = -RHS / (kx² + ky² + kz²)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    ϕ[1, 1, 1] = 0

    solver.IFFT! * ϕ  # Calculate IFFTˣʸ(ϕ̂) in place.

    # Calculate IDCTᶻ(ϕ̂) in place using the FFT.
    ϕ .*= ω_4Nz⁻
    solver.IFFT_DCT! * ϕ

    nothing
end

function solve_poisson_3d!(solver::PoissonSolverGPU{<:PNN}, grid::RegularCartesianGrid)
    Nx, Ny, Nz, _ = unpack_grid(grid)
    kx², ky², kz² = solver.kx², solver.ky², solver.kz²
    ω_4Ny⁺, ω_4Ny⁻, ω_4Nz⁺, ω_4Nz⁻ = solver.ω_4Ny⁺, solver.ω_4Ny⁻, solver.ω_4Nz⁺, solver.ω_4Nz⁻
    r_y_inds, r_z_inds = solver.r_y_inds, solver.r_z_inds
    M_ky, M_kz = solver.M_ky, solver.M_kz

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    B = solver.storage2  # Complex buffer storage.

    # Calculate DCTʸᶻ(f) in place using the FFT.
    solver.FFT_DCT! * RHS

    RHS⁻ = view(RHS, 1:Nx, r_y_inds, 1:Nz)
    @. B = 2 * real(ω_4Nz⁺ * (ω_4Ny⁺ * RHS + ω_4Ny⁻ * RHS⁻))

    solver.FFT! * B # Calculate FFTˣ(f) in place.

    @. B = -B / (kx² + ky² + kz²)

    B[1, 1, 1] = 0  # Setting DC component of the solution (the mean) to be zero.

    solver.IFFT! * B  # Calculate IFFTˣ(ϕ̂) in place.

    # Calculate IDCTʸᶻ(ϕ̂) in place using the FFT.
    B⁻⁺ = view(B, 1:Nx, r_y_inds, 1:Nz)
    B⁺⁻ = view(B, 1:Nx, 1:Ny, r_z_inds)
    B⁻⁻ = view(B, 1:Nx, r_y_inds, r_z_inds)

    @. ϕ = 1/4 *  ω_4Ny⁻ * ω_4Nz⁻ * ((B - M_ky * M_kz * B⁻⁻) - im*(M_kz * B⁺⁻ + M_ky * B⁻⁺))

    solver.IFFT_DCT! * ϕ

    nothing
end
