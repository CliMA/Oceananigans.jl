using Oceananigans
using Oceananigans.Architectures: child_architecture, device
using Oceananigans.Operators: ∇²ᶜᶜᶜ
using KernelAbstractions: @kernel, @index, Event
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: FFTBasedPoissonSolver, solve!, PreconditionedConjugateGradientSolver, MultigridSolver, finalize_solver!
using Oceananigans.Architectures: architecture, arch_array
using IterativeSolvers
using Statistics: mean
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

N = 8
grid = RectilinearGrid(GPU(), size=(N, N), x=(-4, 4), y=(-4, 4), topology=(Bounded, Bounded, Flat))

arch = architecture(grid)

# Select RHS
r = CenterField(grid)
Nx, Ny, Nz = size(r)

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


# Solve ∇²φ = r with `AlgebraicMultigrid` solver
φ_mg = CenterField(grid)

@info "Solving the Poisson equation with the Algebraic Multigrid solver..."
@time mgs = MultigridSolver(compute_∇²!, arch, grid; template_field = r)
@time solve!(φ_mg, mgs, r)
fill_halo_regions!(φ_mg)
φ_mg .-= mean(φ_mg)

finalize_solver!(mgs)



# Solve ∇²φ = r with `PreconditionedConjugateGradientSolver` solver using the AlgebraicMultigrid as preconditioner

struct MultigridPreconditioner{S}
    multigrid_solver :: S
end

mgs = MultigridSolver(compute_∇²!, arch, grid; template_field = r, maxiter = 5)

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
finalize_solver!(mgs)

@show φ_fft
@show φ_cg
@show φ_mg
@show φ_cgmg
