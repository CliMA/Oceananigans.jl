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

import Oceananigans.Solvers: precondition!

N = 64

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


# Solve ∇²φ = r with `PreconditionedConjugateGradientSolver` solver using the in place AlgebraicMultigrid as preconditioner

struct MultigridPreconditionerInplace{S}
    multigrid_solver :: S
end

mgs_inplace = MultigridSolver(compute_∇²!, arch, grid; template_field = r, maxiter = 5, amg_algorithm = RugeStubenAMG())

mgp_inplace = MultigridPreconditionerInplace(mgs_inplace)

"""
    precondition!(z, mgp::MultigridPreconditionerInplace, r, args...)

Return `z` (Field)
"""
function precondition!(z, mgp::MultigridPreconditionerInplace, r, args...)
    solve!(z, mgp.multigrid_solver, r)
    fill_halo_regions!(z)
    println("preconditioning")
    return z
end


φ_inplace = CenterField(grid)
inplace_solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=r, reltol=eps(eltype(grid)), preconditioner = mgp_inplace)


@info "Solving the Poisson equation with provided initial guess..."
@time solve!(φ_inplace, inplace_solver, r, arch, grid)

fill_halo_regions!(φ_inplace)

∇²φ_inplace = CenterField(grid)

compute_∇²!(∇²φ_inplace, φ_inplace, arch, grid)
@show maximum(interior(∇²φ_inplace) - interior(r))



# Solve ∇²φ = r with `PreconditionedConjugateGradientSolver` solver using the zeroed AlgebraicMultigrid as preconditioner

struct MultigridPreconditionerZeroed{S}
    multigrid_solver :: S
end

mgs_zeroed = MultigridSolver(compute_∇²!, arch, grid; template_field = r, maxiter = 5, amg_algorithm = RugeStubenAMG())

mgp_zeroed = MultigridPreconditionerZeroed(mgs_zeroed)

"""
    precondition!(z, mgp::MultigridPreconditionerZeroed, r, args...)

Return `z` (Field)
"""
function precondition!(z, mgp::MultigridPreconditionerZeroed, r, args...)
    z .= 0
    solve!(z, mgp.multigrid_solver, r)
    fill_halo_regions!(z)
    println("preconditioning")
    return z
end


φ_zeroed = CenterField(grid)
zeroed_solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=r, reltol=eps(eltype(grid)), preconditioner = mgp_zeroed)

@info "Solving the Poisson equation with zeroed initial guess..."
@time solve!(φ_zeroed, zeroed_solver, r, arch, grid)

fill_halo_regions!(φ_zeroed)

∇²φ_zeroed = CenterField(grid)

compute_∇²!(∇²φ_zeroed, φ_zeroed, arch, grid)
@show maximum(interior(∇²φ_zeroed) - interior(r))