using Oceananigans

using Oceananigans.Architectures: child_architecture, device, arch_array
using Oceananigans.Operators: ∇²ᶜᶜᶜ
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: solve!,
                            FFTBasedPoissonSolver,
                            PreconditionedConjugateGradientSolver,
                            MultigridSolver,
                            finalize_solver!

using BenchmarkTools,
      CUDA,
      IterativeSolvers,
      GLMakie

using KernelAbstractions: @kernel, @index, Event
using Statistics: mean

import Oceananigans.Solvers: precondition!

"""
Demonstrates how one can solve a Poisson problem using:
  * the FFT Solver,
  * the Conjugate Gradient Solver,
  * the Multigrid Solver, and
  * the Conjugate Gradient Solver preconditioned with the Multigrid Solver.

Also benchmarks the various solvers.
"""

arch = CPU()

Nx, Ny, Nz = 24, 24, 12

z_slice = 6

underlying_grid = RectilinearGrid(arch,
                                  size =(Nx, Ny, Nz),
                                  x = (-4, 4),
                                  y = (-4, 4),
                                  z = (-4, 4),
                                  topology=(Bounded, Bounded, Bounded))

land(x, y) = (abs(x) > 2 && abs(y) > 2) ? 0.8 : 0

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(land))

x, y = grid.underlying_grid.xᶜᵃᵃ, grid.underlying_grid.yᵃᶜᵃ


# Select RHS
r = CenterField(grid)
r₀(x, y, z) = (x^2 + y^2) < 1 ? 1 : 0
# r₀(x, y, z) = exp(-x^2 - y^2 - z^2)
set!(r, r₀)
r .-= mean(r)
fill_halo_regions!(r)

fig = Figure()
ax = Axis(fig[1, 2]; title="right-hand-side")
heatmap!(ax, x[1:Nx], y[1:Ny], interior(r)[:, :, z_slice])
current_figure()

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

# Solve ∇²φ = r with `PreconditionedConjugateGradientSolver`
φ_cg = CenterField(grid)
cg_solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=r, reltol=eps(eltype(grid)))

@info "Solving the Poisson equation with a conjugate gradient iterative solver..."
@time solve!(φ_cg, cg_solver, r, arch, grid.underlying_grid)

fill_halo_regions!(φ_cg)

ax_cg = Axis(fig[2, 1]; title="cg")
heatmap!(ax_cg, x[1:Nx], y[1:Ny], interior(φ_cg)[:, :, z_slice])
current_figure()

function bench_solve()
    parent(φ_cg) .= 0
    solve!(φ_cg, cg_solver, r, arch, grid)
    return nothing
end

if arch == CPU()
    bench_pcg = @benchmark bench_solve();
elseif arch == GPU()
    bench_pcg = @benchmark CUDA.@sync bench_solve();
end

fill_halo_regions!(φ_cg)

# Solve ∇²φ = r with `AlgebraicMultigrid` solver
φ_mg = CenterField(grid)

@info "Constructing an Algebraic Multigrid solver..."
@time mgs = MultigridSolver(compute_∇²!, arch, grid.underlying_grid; template_field = r)

@info "Solving the Poisson equation with the Algebraic Multigrid solver..."
@time solve!(φ_mg, mgs, r)

function bench_solve()
    parent(φ_mg) .= 0
    solve!(φ_mg, mgs, r)
    return nothing
end

if arch == CPU()
    bench_mg = @benchmark bench_solve();
elseif arch == GPU()
    bench_mg = @benchmark CUDA.@sync bench_solve();
end

φ_mg .-= mean(φ_mg)

fill_halo_regions!(φ_mg)
finalize_solver!(mgs)

ax_mg = Axis(fig[2, 2]; title="mg")
heatmap!(ax_mg, x[1:Nx], y[1:Ny], interior(φ_mg)[:, :, z_slice])
current_figure()

# Solve ∇²φ = r with `PreconditionedConjugateGradientSolver` solver using the AlgebraicMultigrid as preconditioner

"""
    struct MultigridPreconditioner{S}

A multigrid preconditioner.
"""
struct MultigridPreconditioner{S}
    multigrid_solver :: S
end

"""
    precondition!(z, mgp::MultigridPreconditioner, r, args...)

Return `z` (Field)
"""
function precondition!(z, mg_solver::MultigridSolver, r, args...)
    parent(z) .= 0
    solve!(z, mg_solver, r)

    return z
end

# construct a multigrid solver to use as a preconditioner
maxiter = 1
mgs = MultigridSolver(compute_∇²!, arch, grid; template_field = r, maxiter)

φ_cgmg = CenterField(grid)

@info "Constructing an Preconditioned Congjugate Gradient solver with Algebraic Multigrid preconditioner..."
cgmg_solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=r, reltol=eps(eltype(grid)), preconditioner = mgs)

@info "Solving the Poisson equation with a conjugate gradient preconditioned iterative solver w/ AMG as preconditioner..."
@time solve!(φ_cgmg, cgmg_solver, r, arch, grid)

function bench_solve()
    parent(φ_cgmg) .= 0
    solve!(φ_cgmg, cgmg_solver, r, arch, grid)

    return nothing
end

if arch == CPU()
    bench_mgpcg = @benchmark bench_solve();
elseif arch == GPU()
    bench_mgpcg = @benchmark CUDA.@sync bench_solve();
end

fill_halo_regions!(φ_cgmg)

ax_cgmg = Axis(fig[2, 3]; title="cgmg")
heatmap!(ax_cgmg, x[1:Nx], y[1:Ny], interior(φ_cgmg)[:, :, z_slice])
current_figure()

finalize_solver!(cgmg_solver)
