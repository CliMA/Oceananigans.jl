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
using BenchmarkTools

import Oceananigans.Solvers: precondition!

"""
Demonstrates how one can solve a Poisson problem using the FFT Solver, the Conjugate Gradient Solver, 
the Multigrid Solver and the Conjugate Gradient Solver preconditioned with the Multigrid Solver.
"""

N = 64

grid = RectilinearGrid(size=(N, N), x=(-4, 4), y=(-4, 4), topology=(Bounded, Bounded, Flat))

arch = architecture(grid)
Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

# Select RHS
r = CenterField(grid)
r₀(x, y, z) = (x^2 + y^2) < 1 ? 1 : 0 # exp(-x^2 - y^2)
set!(r, r₀)
r .-= mean(r)
fill_halo_regions!(r, grid.architecture)


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


# Solve ∇²φ = r with `FFTBasedPoissonSolver`
φ_fft = CenterField(grid)
fft_solver = FFTBasedPoissonSolver(grid)
fft_solver.storage .= interior(r)

@info "Solving the Poisson equation with an FFT-based solver..."
@time solve!(φ_fft, fft_solver, fft_solver.storage)

function bench_solve()
    parent(φ_fft) .= 0
    fft_solver.storage .= interior(r)
    solve!(φ_fft, fft_solver, fft_solver.storage)
    return nothing
end

@show @btime bench_solve();

fill_halo_regions!(φ_fft)


# Solve ∇²φ = r with `PreconditionedConjugateGradientSolver`
φ_cg = CenterField(grid)
cg_solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=r, reltol=eps(eltype(grid)))

@info "Solving the Poisson equation with a conjugate gradient iterative solver..."
@time solve!(φ_cg, cg_solver, r, arch, grid)

function bench_solve()
    parent(φ_cg) .= 0
    solve!(φ_cg, cg_solver, r, arch, grid)
    return nothing
end

@show @btime bench_solve();

fill_halo_regions!(φ_cg)


# Solve ∇²φ = r with `AlgebraicMultigrid` solver
φ_mg = CenterField(grid)

@info "Constructing an Algebraic Multigrid solver..."
@time mgs = MultigridSolver(compute_∇²!, arch, grid; template_field = r)

@info "Solving the Poisson equation with the Algebraic Multigrid solver..."
@time solve!(φ_mg, mgs, r)

function bench_solve()
    parent(φ_mg) .= 0
    solve!(φ_mg, mgs, r)
    return nothing
end

@show @btime bench_solve();

φ_mg .-= mean(φ_mg)

fill_halo_regions!(φ_mg)


# Solve ∇²φ = r with `PreconditionedConjugateGradientSolver` solver using the AlgebraicMultigrid as preconditioner

"""
    struct MultigridPreconditioner{S}

A multigrid preconditioner.
"""
struct MultigridPreconditioner{S}
    multigrid_solver :: S
end

"""
    MultigridPreconditioner(linear_opearation::Function, arch, grid, template_field; maxiter=1)

Return a multigrid preconditioner with maximum iterations: `maxiter`.
"""
function MultigridPreconditioner(linear_opearation::Function, arch, grid, template_field; maxiter=1)
    mgs = MultigridSolver(linear_opearation, arch, grid; template_field, maxiter, amg_algorithm = RugeStubenAMG())
    
    S = typeof(mgs)
    return MultigridPreconditioner{S}(mgs)
end

"""
    precondition!(z, mgp::MultigridPreconditioner, r, args...)
Return `z` (Field)
"""
function precondition!(z, mgp::MultigridPreconditioner, r, args...)
    parent(z) .= 0
    solve!(z, mgp.multigrid_solver, r)

    return z
end

maxiter = 1;
mgp = MultigridPreconditioner(compute_∇²!, arch, grid, r; maxiter)

φ_cgmg = CenterField(grid)

@info "Constructing an Preconditioned Congjugate Gradient solver with Algebraic Multigrid preconditioner..."
cgmg_solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=r, reltol=eps(eltype(grid)), preconditioner = mgp)

@info "Solving the Poisson equation with a conjugate gradient preconditioned iterative solver w/ AMG as preconditioner..."
@time solve!(φ_cgmg, cgmg_solver, r, arch, grid)

function bench_solve()
    parent(φ_cgmg) .= 0
    solve!(φ_cgmg, cgmg_solver, r, arch, grid)

    return nothing
end

@btime bench_solve();

fill_halo_regions!(φ_cgmg)

#=
using  GLMakie
∇²φ_fft = CenterField(grid)
∇²φ_cg = CenterField(grid)
∇²φ_mg = CenterField(grid)
∇²φ_cgmg = CenterField(grid)

compute_∇²!(∇²φ_fft, φ_fft, arch, grid)
compute_∇²!(∇²φ_cg, φ_cg, arch, grid)
compute_∇²!(∇²φ_mg, φ_mg, arch, grid)
compute_∇²!(∇²φ_cgmg, φ_cgmg, arch, grid)

# Plot results
fig = Figure(resolution=(1600, 1200))

ax_r = Axis(fig[1, 3], aspect=1, title="RHS")

ax_φ_fft = Axis(fig[2, 1], aspect=1, title="FFT-based solution")
ax_φ_cg = Axis(fig[2, 3], aspect=1, title="CG solution")
ax_φ_mg = Axis(fig[2, 5], aspect=1, title="Multigrid solution")
ax_φ_cgmg = Axis(fig[2, 7], aspect=1, title="PreconditionedCG MG solution")

ax_∇²φ_fft = Axis(fig[3, 1], aspect=1, title="FFT-based ∇²φ")
ax_∇²φ_cg = Axis(fig[3, 3], aspect=1, title="CG ∇²φ")
ax_∇²φ_mg = Axis(fig[3, 5], aspect=1, title="Multigrid ∇²φ")
ax_∇²φ_cgmg = Axis(fig[3, 7], aspect=1, title="PreconditionedCG MG ∇²φ")

ax_err_fft = Axis(fig[4, 1], aspect=1, title="error FFT-based")
ax_err_cg = Axis(fig[4, 3], aspect=1, title="error CG")
ax_err_mg = Axis(fig[4, 5], aspect=1, title="error Multigrid")
ax_err_cgmg = Axis(fig[4, 7], aspect=1, title="error PreconditionedCG MG")

hm_r = heatmap!(ax_r, interior(r, :, :, 1))
Colorbar(fig[1, 4], hm_r)

hm_fft = heatmap!(ax_φ_fft, interior(φ_fft, :, :, 1))
Colorbar(fig[2, 2], hm_fft)
hm_cg = heatmap!(ax_φ_cg, interior(φ_cg, :, :, 1))
Colorbar(fig[2, 4], hm_cg)
hm_mg = heatmap!(ax_φ_mg, interior(φ_mg, :, :, 1))
Colorbar(fig[2, 6], hm_mg)
hm_cgmg = heatmap!(ax_φ_cgmg, interior(φ_cgmg, :, :, 1))
Colorbar(fig[2, 8], hm_cgmg)

hm_∇²fft = heatmap!(ax_∇²φ_fft, interior(∇²φ_fft, :, :, 1))
Colorbar(fig[3, 2], hm_∇²fft)
hm_∇²cg = heatmap!(ax_∇²φ_cg, interior(∇²φ_cg, :, :, 1))
Colorbar(fig[3, 4], hm_∇²cg)
hm_∇²mg = heatmap!(ax_∇²φ_mg, interior(∇²φ_mg, :, :, 1))
Colorbar(fig[3, 6], hm_∇²mg)
hm_∇²cgmg = heatmap!(ax_∇²φ_cgmg, interior(∇²φ_cgmg, :, :, 1))
Colorbar(fig[3, 8], hm_∇²cgmg)

hm_err_cg = heatmap!(ax_err_cg, interior(∇²φ_cg, :, :, 1) - interior(∇²φ_fft, :, :, 1))
Colorbar(fig[4, 4], hm_err_cg)
hm_err_mg = heatmap!(ax_err_mg, interior(∇²φ_mg, :, :, 1) - interior(∇²φ_fft, :, :, 1))
Colorbar(fig[4, 6], hm_err_mg)
hm_err_cgmg = heatmap!(ax_err_cgmg, interior(∇²φ_cgmg, :, :, 1) - interior(∇²φ_fft, :, :, 1))
Colorbar(fig[4, 8], hm_err_cgmg)

display(fig)
=#