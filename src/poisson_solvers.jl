import FFTW
using GPUifyLoops

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

function PoissonSolverGPU(g::Grid, exfield::CellField)
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

    print("Creating CuFFT plans...\n")
    print("FFT_xy!:  "); @time FFT_xy!  = plan_fft!(exfield.data, [1, 2])
    print("FFT_z!:   "); @time FFT_z!   = plan_fft!(exfield.data, 3)
    print("IFFT_xy!: "); @time IFFT_xy! = plan_ifft!(exfield.data, [1, 2])
    print("IFFT_z!:  "); @time IFFT_z!  = plan_ifft!(exfield.data, 3)

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

    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) f2ϕ!(Val(:GPU), g.Nx, g.Ny, g.Nz, f.data, ϕ.data, solver.kx², solver.ky², solver.kz²)
    ϕ.data[1, 1, 1] = 0

    solver.IFFT_xy! * ϕ.data  # Calculate IFFTˣʸ(ϕ̂) in place.

    # Calculate IDCTᶻ(ϕ̂) in place using the FFT.
    ϕ.data .*= solver.idct_bfactors
    solver.IFFT_z! * ϕ.data
    nothing
end

"Kernel for computing the solution `ϕ` to Poisson equation for source term `f` on a GPU."
function f2ϕ!(::Val{Dev}, Nx, Ny, Nz, f, ϕ, kx², ky², kz²) where Dev
    @setup Dev

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds ϕ[i, j, k] = -f[i, j, k] / (kx²[i] + ky²[j] + kz²[k])
            end
        end
    end

    @synchronize
end


#
# Additional solver functions
#

function solve_poisson_3d_ppn(f, Nx, Ny, Nz, Δx, Δy, Δz)
    Lx, Ly, Lz = Nx*Δx, Ny*Δy, Nz*Δz

    function mkwaves(N,L)
        k²_cyc = zeros(N, 1)
        k²_neu = zeros(N, 1)

        for i in 1:N
            k²_cyc[i] = (2sin((i-1)*π/N)   /(L/N))^2
            k²_neu[i] = (2sin((i-1)*π/(2N))/(L/N))^2
        end

        return k²_cyc, k²_neu
    end

    fh = FFTW.fft(FFTW.r2r(f, FFTW.REDFT10, 3), [1, 2])

    kx²_cyc, kx²_neu = mkwaves(Nx, Lx)
    ky²_cyc, ky²_neu = mkwaves(Ny, Ly)
    kz²_cyc, kz²_neu = mkwaves(Nz, Lz)

    kx² = kx²_cyc
    ky² = ky²_cyc
    kz² = kz²_neu

    ϕh = zeros(Complex{Float64}, Nx, Ny, Nz)

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        @inbounds ϕh[i, j, k] = -fh[i, j, k] / (kx²[i] + ky²[j] + kz²[k])
    end
    ϕh[1, 1, 1] = 0

    FFTW.r2r(real.(FFTW.ifft(ϕh, [1, 2])), FFTW.REDFT01, 3) / (2Nz)
end

function solve_poisson_3d_ppn!(g::RegularCartesianGrid, f::CellField, ϕ::CellField)
    kx² = zeros(g.Nx, 1)
    ky² = zeros(g.Ny, 1)
    kz² = zeros(g.Nz, 1)

    for i in 1:g.Nx; kx²[i] = (2sin((i-1)*π/g.Nx)    / (g.Lx/g.Nx))^2; end
    for j in 1:g.Ny; ky²[j] = (2sin((j-1)*π/g.Ny)    / (g.Ly/g.Ny))^2; end
    for k in 1:g.Nz; kz²[k] = (2sin((k-1)*π/(2g.Nz)) / (g.Lz/g.Nz))^2; end

    FFTW.r2r!(f.data, FFTW.REDFT10, 3)
    FFTW.fft!(f.data, [1, 2])

    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds ϕ.data[i, j, k] = -f.data[i, j, k] / (kx²[i] + ky²[j] + kz²[k])
    end
    ϕ.data[1, 1, 1] = 0

    FFTW.ifft!(ϕ.data, [1, 2])

    @. ϕ.data = real(ϕ.data) / (2g.Nz)

    FFTW.r2r!(ϕ.data, FFTW.REDFT01, 3)

    nothing
end

function dct_dim3_gpu!(g, f, dct_factors)
    # Nx, Ny, Nz = size(f)
    # This is now done in time_step_kernel_part3!.
    # f .= cat(f[:, :, 1:2:g.Nz], f[:, :, g.Nz:-2:2]; dims=3)
    fft!(f, 3)

    # factors = 2 * exp.(collect(-1im*π*(0:Nz-1) / (2*Nz)))
    # f .*= cu(repeat(reshape(factors, 1, 1, Nz), Nx, Ny, 1))
    f .*= dct_factors

    nothing
end

function idct_dim3_gpu!(g, f, idct_bfactors)
    # Nx, Ny, Nz = size(f)

    # bfactors = exp.(collect(1im*π*(0:Nz-1) / (2*Nz)))
    # bfactors[1] *= 0.5

    # f .*= cu(repeat(reshape(bfactors, 1, 1, Nz), Nx, Ny, 1))

    f .*= idct_bfactors
    ifft!(f, 3)

    # Both these steps have been merged into idct_permute! in the time-stepping loop.
    # f .= CuArray{eltype(f)}(reshape(permutedims(cat(f[:, :, 1:Int(g.Nz/2)], f[:, :, end:-1:Int(g.Nz/2)+1]; dims=4), (1, 2, 4, 3)), g.Nx, g.Ny, g.Nz))
    # @. f = real(f)  # Don't do it here. We'll do it when assigning real(ϕ) to pNHS to save some measly FLOPS.

    nothing
end

function solve_poisson_3d_ppn_gpu!(g::RegularCartesianGrid, f::CellField, ϕ::CellField)
    kx² = cu(zeros(g.Nx, 1))
    ky² = cu(zeros(g.Ny, 1))
    kz² = cu(zeros(g.Nz, 1))

    for i in 1:g.Nx; kx²[i] = (2sin((i-1)*π/g.Nx)    / (g.Lx/g.Nx))^2; end
    for j in 1:g.Ny; ky²[j] = (2sin((j-1)*π/g.Ny)    / (g.Ly/g.Ny))^2; end
    for k in 1:g.Nz; kz²[k] = (2sin((k-1)*π/(2g.Nz)) / (g.Lz/g.Nz))^2; end

    print("FFT!  "); @time fft!(f.data, [1, 2])
    print("DCT!  "); @time dct_dim3_gpu!(f.data)

    print("ϕCALC ");
    @time begin
        for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
            @inbounds ϕ.data[i, j, k] = -f.data[i, j, k] / (kx²[i] + ky²[j] + kz²[k])
        end
        ϕ.data[1, 1, 1] = 0
    end

    print("IFFT! "); @time ifft!(ϕ.data, [1, 2])

    @. ϕ.data = real(ϕ.data) / (2g.Nz)
    # for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
    #     ϕ[i, j, k] = real(ϕ[i, j, k])
    # end

    print("IDCT! "); @time idct_dim3_gpu!(f.data)

    nothing
end

function solve_poisson_3d_ppn_gpu!(Tx, Ty, Bx, By, Bz, g::RegularCartesianGrid, f::CellField, ϕ::CellField, kx², ky², kz², dct_factors, idct_bfactors)
    dct_dim3_gpu!(g, f.data, dct_factors)
    @. f.data = real(f.data)

    fft!(f.data, [1, 2])

    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) f2ϕ!(Val(:GPU), g.Nx, g.Ny, g.Nz, f.data, ϕ.data, kx², ky², kz²)
    ϕ.data[1, 1, 1] = 0

    ifft!(ϕ.data, [1, 2])
    idct_dim3_gpu!(g, ϕ.data, idct_bfactors)

    nothing
end
