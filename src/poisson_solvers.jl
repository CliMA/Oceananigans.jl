import FFTW
import GPUifyLoops: @launch, @loop, @synchronize

abstract type PoissonBCs end
struct PPN <: PoissonBCs end  # Periodic BCs in x,y. Neumann BC in z.
struct PNN <: PoissonBCs end  # Periodic BCs in x. Neumann BC in y,z.

PoissonSolver(::CPU, pbcs::PoissonBCs, grid::Grid) = PoissonSolverCPU(pbcs, grid)
PoissonSolver(::GPU, pbcs::PoissonBCs, grid::Grid) = PoissonSolverGPU(pbcs, grid)

struct PoissonSolverCPU{BC, KT, A, FFTT, DCTT, IFFTT, IDCTT} <: PoissonSolver
unpack_grid(grid::Grid) = grid.Nx, grid.Ny, grid.Nz, grid.Lx, grid.Ly, grid.Lz

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
    bcs::BC
    kx²::AAX
    ky²::AAY
    kz²::AAZ
    storage::AA3
    FFT!::FFTT
    DCT!::DCTT
    IFFT!::IFFTT
    IDCT!::IDCTT
end

function PoissonSolverCPU(pbcs::PoissonBCs, grid::Grid, planner_flag=FFTW.PATIENT)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)

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


    if pbcs == PPN()
        FFT!  = FFTW.plan_fft!(storage, [1, 2]; flags=planner_flag)
        IFFT! = FFTW.plan_ifft!(storage, [1, 2]; flags=planner_flag)
        DCT!  = FFTW.plan_r2r!(storage, FFTW.REDFT10, 3; flags=planner_flag)
        IDCT! = FFTW.plan_r2r!(storage, FFTW.REDFT01, 3; flags=planner_flag)
    elseif pbcs == PNN()
        FFT!  = FFTW.plan_fft!(storage, 1; flags=planner_flag)
        IFFT! = FFTW.plan_ifft!(storage, 1; flags=planner_flag)
        DCT!  = FFTW.plan_r2r!(storage, FFTW.REDFT10, [2, 3]; flags=planner_flag)
        IDCT! = FFTW.plan_r2r!(storage, FFTW.REDFT01, [2, 3]; flags=planner_flag)
    end

    PoissonSolverCPU(pbcs, kx², ky², kz², storage, FFT!, DCT!, IFFT!, IDCT!)
end

"""
    solve_poisson_3d!(solver::PoissonSolverCPU, grid::RegularCartesianGrid)

Solve Poisson equation on a staggered grid (Arakawa C-grid) with with
appropriate boundary conditions as specified by solver.bcs  using planned FFTs
and DCTs. The right-hand-side RHS is stored in solver.storage which the solver
mutates to produce the solution, so it will also be stored in solver.storage.

  Args
  ----
  solver : Poisson solver (CPU)
    grid : solver grid
"""
function solve_poisson_3d!(solver::PoissonSolverCPU{<:PPN}, grid::RegularCartesianGrid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    solver.DCT! * RHS  # Calculate DCTᶻ(f) in place.
    solver.FFT! * RHS  # Calculate FFTˣʸ(f) in place.

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        @inbounds ϕ[i, j, k] = -RHS[i, j, k] / (solver.kx²[i] + solver.ky²[j] + solver.kz²[k])
    end

    ϕ[1, 1, 1] = 0  # Setting DC component of the solution (the mean) to be zero.

    solver.IFFT! * ϕ  # Calculate IFFTˣʸ(ϕ) in place.
    solver.IDCT! * ϕ  # Calculate IDCTᶻ(ϕ) in place.

    @. ϕ = ϕ / (2Nz)  # Must normalize by 2Nz after using FFTW.REDFT.

    nothing
end

function solve_poisson_3d!(solver::PoissonSolverCPU{<:PNN}, grid::RegularCartesianGrid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    solver.DCT! * RHS  # Calculate DCTʸᶻ(f) in place.
    solver.FFT! * RHS  # Calculate FFTˣ(f) in place.

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        @inbounds ϕ[i, j, k] = -RHS[i, j, k] / (solver.kx²[i] + solver.ky²[j] + solver.kz²[k])
    end

    ϕ[1, 1, 1] = 0  # Setting DC component of the solution (the mean) to be zero.

    solver.IFFT! * ϕ  # Calculate IFFTˣ(ϕ) in place.
    solver.IDCT! * ϕ  # Calculate IDCTʸᶻ(ϕ) in place.

    @. ϕ = ϕ / (4Ny*Nz)  # Must normalize by 2Ny*2Nz after using FFTW.REDFT.

    nothing
end

struct PoissonSolverGPU{BC, KT, FTY, FTZ, A, FFT, FFTD, IFFT, IFFTD} <: PoissonSolver
    bcs::BC
    kx²::KT
    ky²::KT
    kz²::KT
    dct_factors_y::FTY
    idct_bfactors_y::FTY
    dct_factors_z::FTZ
    idct_bfactors_z::FTZ
    storage::A
    FFT!::FFT
    FFT_DCT!::FFTD
    IFFT!::IFFT
    IFFT_DCT!::IFFTD
end

function PoissonSolverGPU(pbcs::PoissonBCs, grid::Grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Lx, Ly, Lz = grid.Lx, grid.Ly, grid.Lz

    # Storage for RHS and Fourier coefficients is hard-coded to be Float64 because of precision issues with Float32.
    # See https://github.com/climate-machine/Oceananigans.jl/issues/55
    kx² = zeros(Float64, Nx)
    ky² = zeros(Float64, Ny)
    kz² = zeros(Float64, Nz)

    storage = CuArray(zeros(Complex{Float64}, grid.Nx, grid.Ny, grid.Nz))

    # Creating Arrays for ki² and converting them to CuArrays to avoid scalar operations.
    for i in 1:Nx; kx²[i] = (2sin((i-1)*π/Nx)    / (Lx/Nx))^2; end
    for k in 1:Nz; kz²[k] = (2sin((k-1)*π/(2Nz)) / (Lz/Nz))^2; end

    if pbcs == PPN()
        for j in 1:Ny; ky²[j] = (2sin((j-1)*π/Ny) / (Ly/Ny))^2; end
    elseif pbcs == PNN()
        for j in 1:Ny; ky²[j] = (2sin((j-1)*π/(2Ny)) / (Ly/Ny))^2; end
    end

    kx², ky², kz² = CuArray(kx²), CuArray(ky²), CuArray(kz²)

    # Exponential factors required to calculate the DCT on the GPU.
    factors_y = 2 * exp.(collect(-1im*π*(0:Ny-1) / (2Ny)))
    dct_factors_y = CuArray{Complex{Float64}}(reshape(factors_y, 1, Ny, 1))

    factors_z = 2 * exp.(collect(-1im*π*(0:Nz-1) / (2Nz)))
    dct_factors_z = CuArray{Complex{Float64}}(reshape(factors_z, 1, 1, Nz))

    # "Backward" exponential factors required to calculate the IDCT on the GPU.
    bfactors_y = exp.(collect(1im*π*(0:Ny-1) / (2Ny)))
    bfactors_y[1] *= 0.5  # Zeroth coefficient of FFTW's REDFT01 is not multiplied by 2.
    idct_bfactors_y = CuArray{Complex{Float64}}(reshape(bfactors_y, 1, Ny, 1))

    bfactors_z = exp.(collect(1im*π*(0:Nz-1) / (2Nz)))
    bfactors_z[1] *= 0.5  # Zeroth coefficient of FFTW's REDFT01 is not multiplied by 2.
    idct_bfactors_z = CuArray{Complex{Float64}}(reshape(bfactors_z, 1, 1, Nz))

    if pbcs == PPN()
        FFT!      = plan_fft!(storage, [1, 2])
        FFT_DCT!  = plan_fft!(storage, 3)
        IFFT!     = plan_ifft!(storage, [1, 2])
        IFFT_DCT! = plan_ifft!(storage, 3)
    elseif pbcs == PNN()
        FFT!      = plan_fft!(storage, 1)
        FFT_DCT!  = plan_fft!(storage, [2, 3])
        IFFT!     = plan_ifft!(storage, 1)
        IFFT_DCT! = plan_ifft!(storage, [2, 3])
    end

    PoissonSolverGPU(pbcs, kx², ky², kz², dct_factors_y, idct_bfactors_y,
                     dct_factors_z, idct_bfactors_z, storage,
                     FFT!, FFT_DCT!, IFFT!, IFFT_DCT!)
end

"""
    solve_poisson_3d_ppn_gpu_planned!(args...)

Solve Poisson equation with Periodic, Periodic, Neumann boundary conditions in x, y, z using planned
CuFFTs on a GPU.

  Args
  ----
      Tx, Ty : Thread size in x, y
  Bx, By, Bz : Block size in x, y, z
      solver : PoissonSolverGPU
        grid : solver grid
"""
function solve_poisson_3d!(Tx, Ty, Bx, By, Bz, solver::PoissonSolverGPU{<:PPN}, grid::RegularCartesianGrid)
    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    # Calculate DCTᶻ(f) in place using the FFT.
    solver.FFT_DCT! * RHS
    RHS .*= solver.dct_factors_z
    @. RHS = real(RHS)

    solver.FFT! * RHS  # Calculate FFTˣʸ(f) in place.

    @launch device(GPU()) threads=(Tx, Ty) blocks=(Bx, By, Bz) f2ϕ!(grid, RHS, ϕ, solver.kx², solver.ky², solver.kz²)

    ϕ[1, 1, 1] = 0  # Setting DC component of the solution (the mean) to be zero.

    solver.IFFT! * ϕ  # Calculate IFFTˣʸ(ϕ̂) in place.

    # Calculate IDCTᶻ(ϕ̂) in place using the FFT.
    ϕ .*= solver.idct_bfactors_z
    solver.IFFT_DCT! * ϕ

    nothing
end

function solve_poisson_3d!(Tx, Ty, Bx, By, Bz, solver::PoissonSolverGPU{<:PNN}, grid::RegularCartesianGrid)
    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    # Calculate DCTʸᶻ(f) in place using the FFT.
    solver.FFT_DCT! * RHS
    RHS .*= solver.dct_factors_z
    RHS .*= solver.dct_factors_y
    @. RHS = real(RHS)

    solver.FFT! * RHS  # Calculate FFTˣ(f) in place.

    @launch device(GPU()) threads=(Tx, Ty) blocks=(Bx, By, Bz) f2ϕ!(grid, RHS, ϕ, solver.kx², solver.ky², solver.kz²)

    ϕ[1, 1, 1] = 0  # Setting DC component of the solution (the mean) to be zero.

    solver.IFFT! * ϕ  # Calculate IFFTˣ(ϕ̂) in place.

    # Calculate IDCTʸᶻ(ϕ̂) in place using the FFT.
    ϕ .*= solver.idct_bfactors_z
    ϕ .*= solver.idct_bfactors_y
    solver.IFFT_DCT! * ϕ

    nothing
end

"Kernel for computing the solution `ϕ` to Poisson equation for source term `f` on a GPU."
function f2ϕ!(grid::Grid, f, ϕ, kx², ky², kz²)
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds ϕ[i, j, k] = -f[i, j, k] / (kx²[i] + ky²[j] + kz²[k])
            end
        end
    end
    @synchronize
end
