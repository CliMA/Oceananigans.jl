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

N = 150

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

struct MultigridPreconditionerZeroed{S}
    multigrid_solver :: S
end

mgs = MultigridSolver(compute_∇²!, arch, grid; template_field = r, maxiter = 1, amg_algorithm = RugeStubenAMG())

mgp = MultigridPreconditionerZeroed(mgs)

"""
    precondition!(z, mgp::MultigridPreconditionerZeroed, r, args...)
Return `z` (Field)
"""
function precondition!(z, mgp::MultigridPreconditionerZeroed, r, args...)
    parent(z) .= 0
    solve!(z, mgp.multigrid_solver, r)
    fill_halo_regions!(z)
    println("norm(Az-r): ", norm(mgs.matrix * reshape(interior(z), Nx * Ny * Nz) - reshape(interior(r), Nx * Ny * Nz)))
    return z
end

φ = CenterField(grid)

Nx, Ny, Nz = size(r)
abstol = norm(mgs.matrix * reshape(interior(φ), Nx * Ny * Nz) - reshape(interior(r), Nx * Ny * Nz)) * eps(eltype(grid))

solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=r, reltol=eps(eltype(grid)), abstol=abstol, preconditioner = mgp)

@info "Solving the Poisson equation..."
@time solve!(φ, solver, r, arch, grid)

fill_halo_regions!(φ)

∇²φ = CenterField(grid)

compute_∇²!(∇²φ, φ, arch, grid)
@show maximum(interior(∇²φ) - interior(r))
