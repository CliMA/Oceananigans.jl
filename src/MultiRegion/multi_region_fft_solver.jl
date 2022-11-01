using Oceananigans.Architectures: device_event
using FFTW
using CUDA, CUDA.CUFFT
using Oceananigans.Operators

import Oceananigans.Architectures: architecture
import Oceananigans.Models.NonhydrostaticModels: PressureSolver, solve_for_pressure!
import Oceananigans.Solvers: FFTBasedPoissonSolver, solve!

using Oceananigans.Solvers: poisson_eigenvalues, plan_transforms

struct MultiRegionFFTBasedPoissonSolver{G, S}
    grid :: G
    solver :: S
end

PressureSolver(arch, grid::MultiRegionGrid) = MultiRegionFFTBasedPoissonSolver(grid, FFTBasedPoissonSolver(grid))

function FFTBasedPoissonSolver(grid::MultiRegionGrid, planner_flag=FFTW.PATIENT)
    topo = (TX, TY, TZ) =  topology(grid)

    global_grid = reconstruct_global_grid(grid)

    λx = poisson_eigenvalues(global_grid.Nx, global_grid.Lx, 1, TX())
    λy = poisson_eigenvalues(global_grid.Ny, global_grid.Ly, 2, TY())
    λz = poisson_eigenvalues(global_grid.Nz, global_grid.Lz, 3, TZ())

    arch = architecture(grid)

    eigenvalues = (λx = arch_array(arch, λx),
                   λy = arch_array(arch, λy),
                   λz = arch_array(arch, λz))

    storage = unified_array(arch, zeros(complex(eltype(global_grid)), size(global_grid)...))

    # Permutation on the grid will go here!
    transforms = plan_transforms(global_grid, storage, planner_flag)

    # Need buffer for index permutations and transposes.
    buffer_needed = arch isa GPU && Bounded in topo
    buffer = buffer_needed ? similar(storage) : nothing

    return FFTBasedPoissonSolver(global_grid, eigenvalues, storage, buffer, transforms)
end

"""
    solve!(ϕ, solver::FFTBasedPoissonSolver, b, m=0)

Solves the "generalized" Poisson equation,

```math
(∇² + m) ϕ = b,
```

where ``m`` is a number, using a eigenfunction expansion of the discrete Poisson operator
on a staggered grid and for periodic or Neumann boundary conditions.

In-place transforms are applied to ``b``, which means ``b`` must have complex-valued
elements (typically the same type as `solver.storage`).

!!! info "Alternative names for "generalized" Poisson equation
    Equation ``(∇² + m) ϕ = b`` is sometimes called the "screened Poisson" equation
    when ``m < 0``, or the Helmholtz equation when ``m > 0``.
"""

function solve_for_pressure!(pressure, multi_solver::MultiRegionFFTBasedPoissonSolver, Δt, U★)

    solver = multi_solver.solver

    # Calculate right hand side:
    rhs  = multi_solver.solver.storage
    arch = architecture(solver)
    grid = multi_solver.grid

    @apply_regionally unified_pressure_source_term_fft_based_solver!(rhs, Δt, U★, arch, grid, Iterate(1:length(grid)), grid.partition)

    # Solve pressure Poisson given for pressure, given rhs
    solve!(pressure, multi_solver, rhs)

    return nothing
end

function unified_pressure_source_term_fft_based_solver!(rhs, Δt, U★, arch, grid, region, partition)
    rhs_event = launch!(arch, grid, :xyz, _unified_pressure_source_term_fft_based_solver!,
                        rhs, grid, Δt, U★, region, partition; dependencies = device_event(arch))

    wait(device(arch), rhs_event)
end

@kernel function _unified_pressure_source_term_fft_based_solver!(rhs, grid, Δt, U★, region, partition)
    i, j, k = @index(Global, NTuple)
    i′, j′, k′ = global_index(i, j, k, grid, region, partition)
    @inbounds rhs[i′, j′, k′] =  divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

function solve!(ϕ, multi_solver::MultiRegionFFTBasedPoissonSolver, b, m=0)
    
    solver = multi_solver.solver
    arch   = architecture(solver)
    grid   = multi_solver.grid

    λx, λy, λz = solver.eigenvalues

    switch_device!(getdevice(λx))

    # Temporarily store the solution in ϕc
    ϕc = solver.storage

    # Transform b *in-place* to eigenfunction space
    [transform!(b, solver.buffer) for transform! in solver.transforms.forward]

    # Solve the discrete screened Poisson equation (∇² + m) ϕ = b.
    @. ϕc = - b / (λx + λy + λz - m)

    # If m === 0, the "zeroth mode" at `i, j, k = 1, 1, 1` is undetermined;
    # we set this to zero by default. Another slant on this "problem" is that
    # λx[1, 1, 1] + λy[1, 1, 1] + λz[1, 1, 1] = 0, which yields ϕ[1, 1, 1] = Inf or NaN.
    m === 0 && CUDA.@allowscalar ϕc[1, 1, 1] = 0

    # Apply backward transforms in order
    [transform!(ϕc, solver.buffer) for transform! in solver.transforms.backward]

    @apply_regionally redistribute_real_component(ϕ, ϕc, arch, grid, Iterate(1:length(grid)), grid.partition)

    return ϕ
end

function redistribute_real_component(ϕ, ϕc, arch, grid, region, partition)
    copy_event = launch!(arch, grid, :xyz, _redistribute_real_component!, ϕ, ϕc, grid, region, partition, dependencies=device_event(arch))
    wait(device(arch), copy_event)
end

@kernel function _redistribute_real_component!(ϕ, ϕc, grid, region, partition)
    i, j, k = @index(Global, NTuple)
    i′, j′, k′ = global_index(i, j, k, grid, region, partition)
    @inbounds ϕ[i, j, k] = real(ϕc[i′, j′, k′])
end