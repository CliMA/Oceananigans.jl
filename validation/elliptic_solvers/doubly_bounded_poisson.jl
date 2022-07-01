using Oceananigans
using Oceananigans.Architectures: child_architecture, device
using Oceananigans.Operators: volume, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶜᶜᵃ, Δyᵃᶜᵃ, Δxᶜᵃᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, ∇²ᶜᶜᶜ
using KernelAbstractions: @kernel, @index, Event
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: FFTBasedPoissonSolver, solve!, HeptadiagonalIterativeSolver, PreconditionedConjugateGradientSolver, MultigridSolver
using Oceananigans.Architectures: architecture, arch_array
using Statistics: mean
using IterativeSolvers
using Statistics: mean
using AlgebraicMultigrid: RugeStubenAMG
using GLMakie
using OffsetArrays

import Oceananigans.Solvers: precondition!

N = 8

grid = RectilinearGrid(size=(N, N), x=(-4, 4), y=(-4, 4), topology=(Bounded, Bounded, Flat))

arch = architecture(grid)
Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

# Select RHS
r = CenterField(grid)
r₀(x, y, z) = (x^2 + y^2) < 1 ? 1 : 0 #exp(-x^2 - y^2)
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

fill_halo_regions!(φ_fft)


# Solve ∇²φ = r with `PreconditionedConjugateGradientSolver`
φ_cg = CenterField(grid)
cg_solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=r, reltol=eps(eltype(grid)))

@info "Solving the Poisson equation with a conjugate gradient iterative solver..."
@time solve!(φ_cg, cg_solver, r, arch, grid)

fill_halo_regions!(φ_cg)


# Solve ∇²φ = r with `AlgebraicMultigrid` solver

@info "Solving the Poisson equation with the Algebraic Multigrid solver..."

mgs = MultigridSolver(compute_∇²!, arch, grid; template_field = r)
φ_mg = CenterField(grid)

solve!(φ_mg, mgs, r)


# Solve ∇²φ = r with `PreconditionedConjugateGradientSolver` solver using the AlgebraicMultigrid as preconditioner

struct MultigridPreconditioner{S}
    multigrid_solver :: S
end

mgs = MultigridSolver(compute_∇²!, arch, grid; template_field = r, maximum_iterations = 5, amg_algorithm = RugeStubenAMG())

mgp = MultigridPreconditioner(mgs)

using AlgebraicMultigrid: solve, init, _solve!

"""
    precondition!(z, mgp::MultigridPreconditioner, r, args...)

Return `z` (Field)
"""
function precondition!(z, mgp::MultigridPreconditioner, r, args...)
    Nx, Ny, Nz = r.grid.Nx, r.grid.Ny, r.grid.Nz
    
    r_array = collect(reshape(interior(r), Nx * Ny * Nz))

    # the non-allocating version of mg solve does not converge
    # when included in the precondition!
    #
    # z_array = collect(reshape(interior(z), Nx * Ny * Nz)) 
    # solver = mgp.multigrid_solver
    # solt = init(solver.amg_algorithm, solver.linear_operator, r_array)
    # _solve!(z_array, solt.ml, solt.b, maxiter=solver.maximum_iterations, abstol = solver.tolerance)

    z_array = solve(mgp.multigrid_solver.linear_operator, r_array, mgp.multigrid_solver.amg_algorithm, maxiter=mgp.multigrid_solver.maximum_iterations)

    interior(z) .= reshape(z_array, Nx, Ny, Nz)
    fill_halo_regions!(z)
    return z
end



φ_cgmg = CenterField(grid)
cgmg_solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=r, reltol=eps(eltype(grid)), preconditioner = mgp)


@info "Solving the Poisson equation with a conjugate gradient preconditioned iterative solver WITH algebraic multigrid as preconditioner..."
@time solve!(φ_cgmg, cgmg_solver, r, arch, grid)

fill_halo_regions!(φ_cgmg)

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
ax_φ_cg = Axis(fig[2, 3], aspect=1, title="PreconditionedCG solution")
ax_φ_mg = Axis(fig[2, 5], aspect=1, title="Multigrid solution")
ax_φ_cgmg = Axis(fig[2, 7], aspect=1, title="PreconditionedCG MG solution")

ax_∇²φ_fft = Axis(fig[3, 1], aspect=1, title="FFT-based ∇²φ")
ax_∇²φ_cg = Axis(fig[3, 3], aspect=1, title="PreconditionedCG ∇²φ")
ax_∇²φ_mg = Axis(fig[3, 5], aspect=1, title="Multigrid ∇²φ")
ax_∇²φ_cgmg = Axis(fig[3, 7], aspect=1, title="PreconditionedCG MG ∇²φ")

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
hm_∇²mg = heatmap!(ax_∇²φ_mg, reshape(∇²φ_mg, (Nx, Ny)))
Colorbar(fig[3, 6], hm_∇²mg)
hm_∇²cgmg = heatmap!(ax_∇²φ_cgmg, reshape(∇²φ_cgmg, (Nx, Ny)))
Colorbar(fig[3, 8], hm_∇²cgmg)

display(fig)
