using Oceananigans
using Oceananigans.Architectures: child_architecture, device
using Oceananigans.Operators: ∇²ᶜᶜᶜ
using KernelAbstractions: @kernel, @index, Event
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: FFTBasedPoissonSolver, solve!, PreconditionedConjugateGradientSolver, MultigridSolver
using Oceananigans.Architectures: architecture, arch_array
using IterativeSolvers
using Statistics: mean
using AlgebraicMultigrid: RugeStubenAMG
using OffsetArrays
using CUDA.CUSPARSE
using CUDA

import Oceananigans.Solvers: precondition!

"""
Testing the AMGX package
"""

@kernel function ∇²!(∇²f, grid, f)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²f[i, j, k] = ∇²ᶜᶜᶜ(i, j, k, grid, f)
end

function compute_∇²!(∇²φ, φ, arch, grid)
    fill_halo_regions!(φ)
    child_arch = child_architecture(arch)
    event = launch!(child_arch, grid, :xyz, ∇²!, ∇²φ, grid, φ, dependencies=Event(device(child_arch)))
    wait(device(child_arch), event)
    fill_halo_regions!(∇²φ)

    return nothing
end

N = 5
grid = RectilinearGrid(CPU(), size=(N, N), x=(-4, 4), y=(-4, 4), topology=(Bounded, Bounded, Flat))

arch = architecture(grid)
Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

# Select RHS
r = CenterField(grid)
r₀(x, y, z) = (x^2 + y^2) < 1 ? 1 : 0 #exp(-x^2 - y^2)
set!(r, r₀)
r .-= mean(r)
fill_halo_regions!(r, grid.architecture)


# Solve ∇²φ = r with `FFTBasedPoissonSolver`
φ_fft = CenterField(grid)
fft_solver = FFTBasedPoissonSolver(grid)
fft_solver.storage .= interior(r)

@info "Solving the Poisson equation with an FFT-based solver..."
@time solve!(φ_fft, fft_solver, fft_solver.storage)

fill_halo_regions!(φ_fft)


# Solve ∇²φ = r with `PreconditionedConjugateGradientSolver`
φ_cg = CenterField(grid)
cg_solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=r, reltol=eps(eltype(grid)))

@info "Solving the Poisson equation with a conjugate gradient iterative solver..."
@time solve!(φ_cg, cg_solver, r, arch, grid)

fill_halo_regions!(φ_cg)


# Solve ∇²φ = r with `AlgebraicMultigrid` solver on CPU
φ_mg = CenterField(grid)

@info "Solving the Poisson equation with the Algebraic Multigrid solver on CPU..."
@time mgs = MultigridSolver(compute_∇²!, arch, grid; template_field = r)
@time solve!(φ_mg, mgs, r)

fill_halo_regions!(φ_mg)


# Solve ∇²φ = r with `AlgebraicMultigrid` solver on GPU
φ_mg_gpu = CenterField(grid)

@info "Solving the Poisson equation with the Algebraic Multigrid solver on GPU..."
using AMGX
AMGX.initialize()
AMGX.initialize_plugins()

# Create arrays on host
Nx, Ny, Nz = size(r)
FT = eltype(r.grid)
b_array = arch_array(arch, zeros(FT, Nx * Ny * Nz))
x_array = arch_array(arch, zeros(FT, Nx * Ny * Nz))

# Configure solver and allocate arrays on device
config = AMGX.Config(Dict("monitor_residual" => 1, "max_iters" => mgs.maxiter, "store_res_history" => 1));
resources = AMGX.Resources(config)
solver = AMGX.Solver(resources, AMGX.dDDI, config)
v = AMGX.AMGXVector(resources, AMGX.dDDI)
x = AMGX.AMGXVector(resources, AMGX.dDDI)
matrix = AMGX.AMGXMatrix(resources, AMGX.dDDI)

b_array .= reshape(interior(r), Nx * Ny * Nz)
AMGX.upload!(v, b_array)

x_array .= reshape(interior(φ_mg), Nx * Ny * Nz)
AMGX.upload!(x, x_array)

cuCSR = CuSparseMatrixCSR(transpose(mgs.matrix))
@inline sub_one(x) = convert(Int32, x-1)
AMGX.upload!(matrix, 
            map(sub_one, cuCSR.rowPtr), # annoyingly arrays need to be 0 indexed rather than 1 indexed
            map(sub_one, cuCSR.colVal),
            cuCSR.nzVal
            )

AMGX.setup!(solver, matrix)
AMGX.solve!(x, solver, v)

interior(φ_mg) .= reshape(Vector(x), Nx, Ny, Nz)

# Free memory
close(matrix); close(x); close(v); close(solver); close(resources); close(config); AMGX.finalize_plugins(); AMGX.finalize()

fill_halo_regions!(φ_mg_gpu)


# Solve ∇²φ = r with `PreconditionedConjugateGradientSolver` solver using the AlgebraicMultigrid as preconditioner

struct MultigridPreconditioner{S}
    multigrid_solver :: S
end

mgs = MultigridSolver(compute_∇²!, arch, grid; template_field = r, maxiter = 5, amg_algorithm = RugeStubenAMG())

mgp = MultigridPreconditioner(mgs)


"""
    precondition!(z, mgp::MultigridPreconditioner, r, args...)

Return `z` (Field)
"""
function precondition!(z, mgp::MultigridPreconditioner, r, args...)
    solve!(z, mgp.multigrid_solver, r)
    fill_halo_regions!(z)
    return z
end


φ_cgmg = CenterField(grid)
cgmg_solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=r, reltol=eps(eltype(grid)), preconditioner = mgp)


@info "Solving the Poisson equation with a conjugate gradient preconditioned iterative solver WITH algebraic multigrid as preconditioner..."
@time solve!(φ_cgmg, cgmg_solver, r, arch, grid)

fill_halo_regions!(φ_cgmg)

@show φ_fft
@show φ_cg
@show φ_mg
@show φ_mg_gpu
@show φ_cgmg