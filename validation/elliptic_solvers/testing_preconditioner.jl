using Oceananigans
using Oceananigans.Architectures: child_architecture, device
using KernelAbstractions: @kernel, @index, Event
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: ∇²ᶜᶜᶜ
using Oceananigans.Solvers: solve!, PreconditionedConjugateGradientSolver, MultigridSolver
using Oceananigans.Architectures: architecture
using Statistics: mean
using IterativeSolvers
using OffsetArrays
using AlgebraicMultigrid: RugeStubenAMG
using LinearAlgebra

import Oceananigans.Solvers: precondition!

N = 256

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

maxiter = 1;
mgp = MultigridPreconditioner(compute_∇²!, arch, grid, r; maxiter)

"""
    precondition!(z, mgp::MultigridPreconditioner, r, args...)

Return `z` (Field)
"""
function precondition!(z, mgp::MultigridPreconditioner, r, args...)
    parent(z) .= 0
    solve!(z, mgp.multigrid_solver, r)

    Nx, Ny, Nz = size(r)

    mgs = mgp.multigrid_solver
    
    println("|Az - r|: ", norm(mgs.matrix * reshape(interior(z), Nx * Ny * Nz) - reshape(interior(r), Nx * Ny * Nz)))

    return z
end

φ = CenterField(grid)

Nx, Ny, Nz = size(r)

# abstol = norm(mgs.matrix * reshape(interior(φ), Nx * Ny * Nz) - reshape(interior(r), Nx * Ny * Nz)) * eps(eltype(grid))
@show abstol = sqrt(eps(eltype(grid)))
@show reltol = 0sqrt(eps(eltype(grid)))

@info "Solving the Poisson with PGC + MG preconditioner..."
solver = PreconditionedConjugateGradientSolver(compute_∇²!; template_field=r, reltol, abstol, preconditioner = mgp)

@time solve!(φ, solver, r, arch, grid)

fill_halo_regions!(φ)

∇²φ = CenterField(grid)

compute_∇²!(∇²φ, φ, arch, grid)
@show maximum(interior(∇²φ) - interior(r))


@info "Solving the Poisson with PGC starting from a good initial guess from a MG solver..."
# Solve ∇²φ = r with `ConjugateGradientSolver` solver using the AlgebraicMultigrid as initial guess
mgs = MultigridSolver(compute_∇²!, arch, grid; template_field = r, maxiter = 5, amg_algorithm = RugeStubenAMG())

φ = CenterField(grid)

Nx, Ny, Nz = size(r)
@show abstol = norm(mgs.matrix * reshape(interior(φ), Nx * Ny * Nz) - reshape(interior(r), Nx * Ny * Nz)) * eps(eltype(grid))
@show reltol = sqrt(eps(eltype(grid)))

solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field = r, reltol = reltol, abstol = abstol)

# @info "Computing a good initial guess..."
solve!(φ, mgs, r)
fill_halo_regions!(φ)

∇²φ = CenterField(grid)

compute_∇²!(∇²φ, φ, arch, grid)
@info "the norm of the initial guess"
@show maximum(interior(∇²φ) - interior(r))

# @info "Solving the Poisson equation with CG solver..."
@time solve!(φ, solver, r, arch, grid)

fill_halo_regions!(φ)

∇²φ = CenterField(grid)

compute_∇²!(∇²φ, φ, arch, grid)
@info "the norm of the PGC solution with good initial guess"
@show maximum(interior(∇²φ) - interior(r))
