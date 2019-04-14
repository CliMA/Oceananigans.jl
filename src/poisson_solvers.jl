import FFTW
import GPUifyLoops: @launch, @loop, @synchronize

# Translations to print FFT timing.
let pf2s = Dict(FFTW.ESTIMATE   => "FFTW.ESTIMATE",
                FFTW.MEASURE    => "FFTW.MEASURE",
                FFTW.PATIENT    => "FFTW.PATIENT",
                FFTW.EXHAUSTIVE => "FFTW.EXHAUSTIVE")
    global plannerflag2string
    plannerflag2string(k::Integer) = pf2s[Int(k)]
end

PoissonSolver(::CPU, grid::Grid) = PoissonSolverCPU(grid)
PoissonSolver(::GPU, grid::Grid) = PoissonSolverGPU(grid)

"""
    PoissonSolver(grid, example_field, planner_flag; verbose=false)

Return a `PoissonSolver` on `grid`, using `example_field` and `planner_flag`
to plan fast transforms.
"""
struct PoissonSolverCPU{T<:AbstractArray} <: PoissonSolver
    kx²::T
    ky²::T
    kz²::T
    storage
    FFT!
    DCT!
    IFFT!
    IDCT!
end

function PoissonSolverCPU(grid::Grid, planner_flag=FFTW.PATIENT)
    T = eltype(grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Lx, Ly, Lz = grid.Lx, grid.Ly, grid.Lz

    # Storage for RHS and Fourier coefficients is hard-coded to be Float64
    # because of precision issues with Float32.
    # See https://github.com/climate-machine/Oceananigans.jl/issues/55
    kx² = zeros(Float64, Nx)
    ky² = zeros(Float64, Ny)
    kz² = zeros(Float64, Nz)

    storage = zeros(Complex{Float64}, grid.Nx, grid.Ny, grid.Nz)

    for i in 1:Nx; kx²[i] = (2sin((i-1)*π/Nx)    / (Lx/Nx))^2; end
    for j in 1:Ny; ky²[j] = (2sin((j-1)*π/Ny)    / (Ly/Ny))^2; end
    for k in 1:Nz; kz²[k] = (2sin((k-1)*π/(2Nz)) / (Lz/Nz))^2; end

    FFT!  = FFTW.plan_fft!(storage, [1, 2]; flags=planner_flag)
    IFFT! = FFTW.plan_ifft!(storage, [1, 2]; flags=planner_flag)
    DCT!  = FFTW.plan_r2r!(storage, FFTW.REDFT10, 3; flags=planner_flag)
    IDCT! = FFTW.plan_r2r!(storage, FFTW.REDFT01, 3; flags=planner_flag)

    PoissonSolverCPU{Array{Float64, 1}}(kx², ky², kz², storage, FFT!, DCT!, IFFT!, IDCT!)
end

"""
    solve_poisson_3d_ppn_planned!(args...)

Solve Poisson equation with Periodic, Periodic, Neumann boundary conditions in x, y, z using planned
FFTs and DCTs.

  Args
  ----

      solver : PoissonSolver
           g : solver grid
           f : RHS to Poisson equation
           ϕ : Solution to Poisson equation
"""
function solve_poisson_3d_ppn_planned!(solver::PoissonSolverCPU, grid::RegularCartesianGrid)
    RHS, ϕ = solver.storage, solver.storage

    solver.DCT! * RHS  # Calculate DCTᶻ(f) in place.
    solver.FFT! * RHS  # Calculate FFTˣʸ(f) in place.

    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        @inbounds ϕ[i, j, k] = -RHS[i, j, k] / (solver.kx²[i] + solver.ky²[j] + solver.kz²[k])
    end

    ϕ[1, 1, 1] = 0  # Setting DC component of the solution (the mean) to be zero.

    solver.IFFT! * ϕ  # Calculate IFFTˣʸ(ϕ) in place.
    solver.IDCT! * ϕ  # Calculate IDCTᶻ(ϕ) in place.

    @. ϕ = ϕ / (2*grid.Nz)  # Must normalize by 2Nz after using FFTW.REDFT.

    nothing
end


"""
    PoissonSolverGPU(grid, example_field)

Return a `PoissonSolverGPU` on `grid`, using `example_field` to plan
CuFFTs on a GPU.
"""
struct PoissonSolverGPU{T<:AbstractArray} <: PoissonSolver
    kx²::T
    ky²::T
    kz²::T
    dct_factors::T
    idct_bfactors::T
    storage
    FFT_xy!
    FFT_z!
    IFFT_xy!
    IFFT_z!
end

function PoissonSolverGPU(g::Grid)
    kx² = CuArray{Float64}(undef, g.Nx)
    ky² = CuArray{Float64}(undef, g.Ny)
    kz² = CuArray{Float64}(undef, g.Nz)

    for i in 1:g.Nx; kx²[i] = (2sin((i-1)*π/g.Nx)    / (g.Lx/g.Nx))^2; end
    for j in 1:g.Ny; ky²[j] = (2sin((j-1)*π/g.Ny)    / (g.Ly/g.Ny))^2; end
    for k in 1:g.Nz; kz²[k] = (2sin((k-1)*π/(2g.Nz)) / (g.Lz/g.Nz))^2; end

    # Exponential factors required to calculate the DCT on the GPU.
    factors = 2 * exp.(collect(-1im*π*(0:g.Nz-1) / (2*g.Nz)))
    dct_factors = CuArray{Complex{Float64}}(repeat(reshape(factors, 1, 1, g.Nz), g.Nx, g.Ny, 1))

    # "Backward" exponential factors required to calculate the IDCT on the GPU.
    bfactors = exp.(collect(1im*π*(0:g.Nz-1) / (2*g.Nz)))
    bfactors[1] *= 0.5
    idct_bfactors = CuArray{Complex{Float64}}(repeat(reshape(bfactors, 1, 1, g.Nz), g.Nx, g.Ny, 1))

    if verbose
        print("Creating CuFFT plans...\n")
        print("FFT_xy!:  "); @time FFT_xy!  = plan_fft!(exfield.data, [1, 2])
        print("FFT_z!:   "); @time FFT_z!   = plan_fft!(exfield.data, 3)
        print("IFFT_xy!: "); @time IFFT_xy! = plan_ifft!(exfield.data, [1, 2])
        print("IFFT_z!:  "); @time IFFT_z!  = plan_ifft!(exfield.data, 3)
    else
        FFT_xy!  = plan_fft!(exfield.data, [1, 2])
        FFT_z!   = plan_fft!(exfield.data, 3)
        IFFT_xy! = plan_ifft!(exfield.data, [1, 2])
        IFFT_z!  = plan_ifft!(exfield.data, 3)
    end

    PoissonSolverGPU{CuArray{Float64}}(kx², ky², kz², dct_factors, idct_bfactors, FFT_xy!, FFT_z!, IFFT_xy!, IFFT_z!)
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
           g : solver grid
           f : RHS to Poisson equation
           ϕ : Solution to Poisson equation
"""
function solve_poisson_3d_ppn_gpu_planned!(Tx, Ty, Bx, By, Bz, solver::PoissonSolverGPU, g::RegularCartesianGrid, f::CellField, ϕ::CellField)
    # Calculate DCTᶻ(f) in place using the FFT.
    solver.FFT_z! * f.data
    f.data .*= solver.dct_factors
    @. f.data = real(f.data)

    solver.FFT_xy! * f.data  # Calculate FFTˣʸ(f) in place.

    @launch device(GPU()) f2ϕ!(g, f.data, ϕ.data, solver.kx², solver.ky², solver.kz², threads=(Tx, Ty), blocks=(Bx, By, Bz))
    ϕ.data[1, 1, 1] = 0

    solver.IFFT_xy! * ϕ.data  # Calculate IFFTˣʸ(ϕ̂) in place.

    # Calculate IDCTᶻ(ϕ̂) in place using the FFT.
    ϕ.data .*= solver.idct_bfactors
    solver.IFFT_z! * ϕ.data
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
