import FFTW

using GPUifyLoops

const solver_eltype = Complex{Float64} # Enforce Float64 for Poisson solver irrespective of grid type.

abstract type PoissonBC end
struct Periodic <: PoissonBC end
struct Neumann <: PoissonBC end

abstract type AbstractPoissonSolver{D, xBC, yBC, zBC} end

struct PoissonEigenvalues{A, xBC, yBC, zBC}
    kx²::A
    ky²::A
    kz²::A
end

get_eigenvalues(::Periodic,  N, L) = [ (2N*sin((i-1)*π/N) / L)^2 for i=1:N]
get_eigenvalues(::Neumann, N, L) = [ (4N*sin((i-1)*π/N) / L)^2 for i=1:N]

function PoissonEigenvalues(A, grid, bcs=(Periodic, Periodic, Neumann))
    kx² = convert(A, reshape(get_eigenvalues(bcs[1], grid.Nx, grid.Lx), (grid.Nx, 1, 1) ))
    ky² = convert(A, reshape(get_eigenvalues(bcs[2], grid.Ny, grid.Ly), (1, grid.Ny, 1) ))
    kz² = convert(A, reshape(get_eigenvalues(bcs[3], grid.Nz, grid.Lz), (1, 1, grid.Nz) ))
    PoissonEigenvalues{A, bcs[1], bcs[2], bcs[3]}(kx², ky², kz²)
end


"""
        PoissonSolver([D=CPU], grid, planning_array=zeros(eltype(grid), size(grid);
                       planner_flags=FFTW.MEASURE, verbose=false, bcs=(Periodic, Periodic, Neumann))

Construct a `PoissonSolver` for solving Poisson's equation on a staggered grid
with 2nd-order finite differences, using the Fast Fourier Transform.
"""
struct PoissonSolver_PPN_CPU{Ak, T1, T2, T3, T4} <: AbstractPoissonSolver{CPU, Periodic, Periodic, Neumann}
    eigenvals :: PoissonEigenvalues{Ak, Periodic, Periodic, Neumann}
    FFT_xy!   :: T1
    DCT_z!    :: T2
    IFFT_xy!  :: T3
    IDCT_z!   :: T4
end

struct PoissonSolver_PPN_GPU{Ak, As, T1, T2, T3, T4} <: AbstractPoissonSolver{GPU, Periodic, Periodic, Neumann}
    eigenvals            :: PoissonEigenvalues{Ak, Periodic, Periodic, Neumann}
    FFT_xy!              :: T1
    FFT_z!               :: T2
    IFFT_xy!             :: T3
    IFFT_z!              :: T4
    fft_to_dct_factors   :: As
    ifft_to_idct_factors :: As
end

get_Ak(::CPU) = Array{solver_eltype,3}
get_Ak(::GPU) = CuArray{solver_eltype,3}

"""
    PoissonSolver(CPU, Periodic, Periodic, Neumann...)

"""
function PoissonSolver(::CPU, ::Periodic, ::Periodic, ::Neumann, grid; 
                       planning_array=zeros(eltype(grid), size(grid)), planner_flag=FFTW.MEASURE, verbose=false)
    Ak = get_Ak(CPU)
    eigenvals = PoissonEigenvalues(Ak, grid, (Periodic, Periodic, Neumann))

     FFT_xy! = plan_fft!( example_array, [1, 2]; flags=planner_flag) 
    IFFT_xy! = plan_ifft!(example_array, [1, 2]; flags=planner_flag) 

     DCT_z! = plan_r2r!(example_array, FFTW.REDFT10, 3; flags=planner_flag) 
    IDCT_z! = plan_r2r!(example_array, FFTW.REDFT01, 3; flags=planner_flag) 
    PoissonSolver_PPN_CPU(eigenvals, FFT_xy!, DCT_z!, IFFT_xy!, DCT_z!)
end

"""
    PoissonSolver(GPU, Periodic, Periodic, Neumann...)

"""
function PoissonSolver(::GPU, ::Periodic, ::Periodic, ::Neumann, grid; 
                       planning_array=CuArray{solver_eltype}(zeros(eltype(grid), size(grid))), 
                       planner_flag=FFTW.MEASURE, verbose=false)
    Ak = get_Ak(GPU)
    eigenvals = PoissonEigenvalues(Ak, grid, (Periodic, Periodic, Neumann))

    FFT_xy!  =  plan_fft!(planning_array, [1, 2])
    FFT_z!   =  plan_fft!(planning_array, 3)
    IFFT_xy! = plan_ifft!(planning_array, [1, 2])
    IFFT_z!  = plan_ifft!(planning_array, 3)

    Nx, Ny, Nz = size(grid)

    # Exponential factors required to calculate the DCT on the GPU.
    fft_to_dct_factors = 2 * exp.(-im*π*(0:Nz-1)/2Nz)
    fft_to_dct_factors = CuArray{solver_eltype}(reshape(fft_to_dct_factors, (1, 1, Nz)))

    # "Backward" exponential factors required to calculate the IDCT on the GPU.
    ifft_to_idct_factors  = exp.(im*π*(0:Nz-1)/2Nz)
    ifft_to_idct_factors = CuArray{solver_eltype}(reshape(ifft_to_idct_factors, (1, 1, Nz)))
    ifft_to_idct_factors[1] *= 0.5

    PoissonSolver_PPN_GPU(eigenvals, FFT_xy!, FFT_z!, IFFT_xy!, IFFT_z!, fft_to_dct_factors, ifft_to_idct_factors)
end

"""
    fwd_transforms!(solver, a)

Perform forward transforms on `a` for `solver`, 
depending on device type and boundary conditions.
"""
function fwd_transforms!(solver::AbstractPoissonSolver{D, xBC, yBC, zBC}, a) where {D<:CPU, xBC<:Periodic, yBC<:Periodic, zBC<:Neumann}
    solver.FFT_xy! * a # Calculate FFTˣʸ(f) in place.
     solver.DCT_z! * a # Calculate DCTᶻ(f) in place.
    return nothing
end

function fwd_transforms!(solver::AbstractPoissonSolver{D, xBC, yBC, zBC}, a) where {D<:GPU, xBC<:Periodic, yBC<:Periodic, zBC<:Neumann}
    # Calculate DCTᶻ(f) in place using the FFT.
    solver.FFT_z! * a
    @. a *= solver.fft_to_dct_factors
    @. a = real(a)

    solver.FFT_xy! * a # Calculate FFTˣʸ(f) in place.

    return nothing
end

function bwd_transforms!(solver::AbstractPoissonSolver{D, xBC, yBC, zBC}, a) where {D<:CPU, xBC<:Periodic, yBC<:Periodic, zBC<:Neumann}
    solver.IFFT!*a # Calculate IFFTˣʸ(ϕ) in place.
    solver.IDCT!*a # Calculate IDCTᶻ(ϕ) in place.
    @. a = a / (2*size(a, 3)) # Specific to PPN boundary conditions?
    return nothing
end

function bwd_transforms!(solver::AbstractPoissonSolver{D, xBC, yBC, zBC}, a) where {D<:GPU, xBC<:Periodic, yBC<:Periodic, zBC<:Neumann}
    solver.IFFT_xy! * a # Calculate IFFTˣʸ(ϕ̂) in place.

    # Calculate IDCTᶻ(ϕ̂) in place using the FFT.
    @. a *= solver.idct_bfactors
    solver.IFFT_z! * a

    return nothing
end
 

"""
    solve_poisson!(ϕ, f, solver)

Solve Poisson's equation for `ϕ`:

``
\triangle \phi = f
``

for `ϕ`, using `solver`.
"""
function solve_poisson!(f, ϕ, solver::PoissonSolver{D, xBC, yBC, zBC}) where {D, xBC<:Periodic, yBC<:Periodic, zBC<:Neumann}
    # Transform source term
    fwd_transforms!(solver, f.data) 

    # Solve Poisson!
    @. ϕ.data = f.data / (kx² + ky² + kz²)
    ϕ.data[1, 1, 1] = 0

    # Transform solution
    bwd_transforms!(solver, ϕ.data)

    return nothing
end
