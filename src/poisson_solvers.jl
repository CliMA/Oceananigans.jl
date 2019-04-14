import FFTW
import GPUifyLoops: @launch, @loop, @synchronize

function init_poisson_solver(::CPU, grid::Grid)
    tmp_rhs .= rand(Float64, grid.Nx, grid.Ny, grid.Nz)
    PoissonSolver(g, tmp_rhs, FFTW.MEASURE)
end

function init_poisson_solver(::GPU, grid::Grid)
    tmp_rhs .= CuArray{Complex{Float64}}(rand(Float64, grid.Nx, grid.Ny, grid.Nz))
    PoissonSolverGPU(g, tmp_rhs)
end

# Translations to print FFT timing.
let pf2s = Dict(FFTW.ESTIMATE   => "FFTW.ESTIMATE",
                FFTW.MEASURE    => "FFTW.MEASURE",
                FFTW.PATIENT    => "FFTW.PATIENT",
                FFTW.EXHAUSTIVE => "FFTW.EXHAUSTIVE")
    global plannerflag2string
    plannerflag2string(k::Integer) = pf2s[Int(k)]
end

"""
    PoissonSolver(grid, example_field, planner_flag; verbose=false)

Return a `PoissonSolver` on `grid`, using `example_field` and `planner_flag`
to plan fast transforms.
"""
struct PoissonSolver{T<:AbstractArray} <: AbstractPoissonSolver
    kx²::T
    ky²::T
    kz²::T
    FFT!
    DCT!
    IFFT!
    IDCT!
end

function PoissonSolver(g::Grid, exfield::CellField, planner_flag=FFTW.PATIENT; verbose=false)
    kx² = zeros(eltype(g), g.Nx)
    ky² = zeros(eltype(g), g.Ny)
    kz² = zeros(eltype(g), g.Nz)

    for i in 1:g.Nx; kx²[i] = (2sin((i-1)*π/g.Nx)    / (g.Lx/g.Nx))^2; end
    for j in 1:g.Ny; ky²[j] = (2sin((j-1)*π/g.Ny)    / (g.Ly/g.Ny))^2; end
    for k in 1:g.Nz; kz²[k] = (2sin((k-1)*π/(2g.Nz)) / (g.Lz/g.Nz))^2; end

    if verbose
        print("Planning Fourier transforms... (planner_flag=$(plannerflag2string(planner_flag)))\n")
        print("FFT!:  "); @time FFT!  = FFTW.plan_fft!(exfield.data, [1, 2]; flags=planner_flag)
        print("IFFT!: "); @time IFFT! = FFTW.plan_ifft!(exfield.data, [1, 2]; flags=planner_flag)
        print("DCT!:  "); @time DCT!  = FFTW.plan_r2r!(exfield.data, FFTW.REDFT10, 3; flags=planner_flag)
        print("IDCT!: "); @time IDCT! = FFTW.plan_r2r!(exfield.data, FFTW.REDFT01, 3; flags=planner_flag)
    else
        FFT!  = FFTW.plan_fft!(exfield.data, [1, 2]; flags=planner_flag)
        IFFT! = FFTW.plan_ifft!(exfield.data, [1, 2]; flags=planner_flag)
        DCT!  = FFTW.plan_r2r!(exfield.data, FFTW.REDFT10, 3; flags=planner_flag)
        IDCT! = FFTW.plan_r2r!(exfield.data, FFTW.REDFT01, 3; flags=planner_flag)
    end

    PoissonSolver{Array{eltype(g),1}}(kx², ky², kz², FFT!, DCT!, IFFT!, IDCT!)
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
function solve_poisson_3d_ppn_planned!(solver::PoissonSolver, g::RegularCartesianGrid, f::CellField, ϕ::CellField)
    solver.DCT!*f.data  # Calculate DCTᶻ(f) in place.
    solver.FFT!*f.data  # Calculate FFTˣʸ(f) in place.

    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds ϕ.data[i, j, k] = -f.data[i, j, k] / (solver.kx²[i] + solver.ky²[j] + solver.kz²[k])
    end
    ϕ.data[1, 1, 1] = 0

    solver.IFFT!*ϕ.data  # Calculate IFFTˣʸ(ϕ) in place.
    solver.IDCT!*ϕ.data  # Calculate IDCTᶻ(ϕ) in place.
    @. ϕ.data = ϕ.data / (2*g.Nz)
    nothing
end


"""
    PoissonSolverGPU(grid, example_field)

Return a `PoissonSolverGPU` on `grid`, using `example_field` to plan
CuFFTs on a GPU.
"""
struct PoissonSolverGPU{T<:AbstractArray} <: AbstractPoissonSolver
    kx²
    ky²
    kz²
    dct_factors
    idct_bfactors
    FFT_xy!
    FFT_z!
    IFFT_xy!
    IFFT_z!
end

function PoissonSolverGPU(g::Grid, exfield::CellField; verbose=false)
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
