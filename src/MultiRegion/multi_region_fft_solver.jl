using Oceananigans.Architectures: device_event
using FFTW
using CUDA, CUDA.CUFFT
using Oceananigans.Operators

import Oceananigans.Architectures: architecture
import Oceananigans.Models.NonhydrostaticModels: PressureSolver, solve_for_pressure!
import Oceananigans.Solvers: FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver, solve!

using Oceananigans.Grids 
using Oceananigans.Solvers: poisson_eigenvalues, plan_transforms

struct MultiRegionPoissonSolver{G, S}
    grid :: G
    solver :: S
end

PressureSolver(arch, grid::RegMultiRegionGrid)  = MultiRegionPoissonSolver(grid, FFTBasedPoissonSolver(grid))
PressureSolver(arch, grid::HRegMultiRegionGrid) = MultiRegionPoissonSolver(grid, FourierTridiagonalPoissonSolver(grid))

function FFTBasedPoissonSolver(grid::MultiRegionGrid, planner_flag=FFTW.PATIENT)
    global_grid = reconstruct_global_grid(grid)

    arch    = architecture(global_grid)
    storage = unified_array(arch, zeros(complex(eltype(global_grid)), size(global_grid)...))
    s       = FFTBasedPoissonSolver(global_grid, planner_flag)

    return FFTBasedPoissonSolver(s.grid, s.eigenvalues, storage, s.buffer, s.transforms)
end

function FourierTridiagonalPoissonSolver(grid::MultiRegionGrid, planner_flag=FFTW.PATIENT)
    global_grid = reconstruct_global_grid(grid)

    arch        = architecture(global_grid)
    storage     = unified_array(arch, zeros(complex(eltype(global_grid)), size(global_grid)...))
    source_term = unified_array(arch, zeros(complex(eltype(global_grid)), size(global_grid)...))
    s           = FourierTridiagonalPoissonSolver(global_grid, planner_flag)

    return FourierTridiagonalPoissonSolver(s.grid, s.batched_tridiagonal_solver, source_term, storage, s.buffer, s.transforms)
end

const MRFFT = MultiRegionPoissonSolver{<:RegMultiRegionGrid}
const MRFTS = MultiRegionPoissonSolver{<:HRegMultiRegionGrid}

function solve_for_pressure!(pressure, multi_solver::MRFFT, Δt, U★)

    solver = multi_solver.solver

    # Calculate right hand side:
    rhs  = multi_solver.solver.storage
    arch = architecture(solver)
    grid = multi_solver.grid

    regions = Iterate(1:length(grid))

    @apply_regionally unified_pressure_source_term_fft_based_solver!(rhs, Δt, U★, arch, grid, regions, grid.partition)

    # Solve pressure Poisson given for pressure, given rhs
    solve!(pressure, multi_solver, rhs)

    return nothing
end

function solve_for_pressure!(pressure, multi_solver::MRFTS, Δt, U★)

    solver = multi_solver.solver

    # Calculate right hand side:
    rhs  = solver.source_term
    arch = architecture(solver)
    grid = multi_solver.grid

    regions = Iterate(1:length(grid))

    @apply_regionally unified_pressure_source_term_fourier_tridiagonal_solver!(rhs, Δt, U★, arch, grid, regions, grid.partition)

    # Solve pressure Poisson given for pressure, given rhs
    solve!(pressure, multi_solver)

    return nothing
end

function unified_pressure_source_term_fft_based_solver!(rhs, Δt, U★, arch, grid, region, partition)
    rhs_event = launch!(arch, grid, :xyz, _unified_pressure_source_term_fft_based_solver!,
                        rhs, grid, Δt, U★, region, partition; dependencies = device_event(arch))

    wait(device(arch), rhs_event)
end

function unified_pressure_source_term_fourier_tridiagonal_solver!(rhs, Δt, U★, arch, grid, region, partition)
    rhs_event = launch!(arch, grid, :xyz, _unified_pressure_source_term_fourier_tridiagonal_solver!,
                        rhs, grid, Δt, U★, region, partition; dependencies = device_event(arch))

    wait(device(arch), rhs_event)
end

@kernel function _unified_pressure_source_term_fft_based_solver!(rhs, grid, Δt, U★, region, partition)
    i, j, k = @index(Global, NTuple)
    i′, j′, k′ = global_index(i, j, k, grid, region, partition)
    @inbounds rhs[i′, j′, k′] =  divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

@kernel function _unified_pressure_source_term_fourier_tridiagonal_solver!(rhs, grid, Δt, U★, region, partition)
    i, j, k = @index(Global, NTuple)
    i′, j′, k′ = global_index(i, j, k, grid, region, partition)
    @inbounds rhs[i′, j′, k′] =  Δzᶜᶜᶜ(i, j, k, grid) * divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

function solve!(ϕ, multi_solver::MRFFT, b, m=0)
    
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

    @apply_regionally redistribute_real_component!(ϕ, ϕc, arch, grid, Iterate(1:length(grid)), grid.partition)

    return ϕ
end

function solve!(x, multi_solver::MRFTS)

    solver = multi_solver.solver
    arch   = architecture(solver)
    grid   = multi_solver.grid

    ϕ = solver.storage

    switch_device!(getdevice(solver.batched_tridiagonal_solver.a))
    
    # Apply forward transforms in order
    [transform!(solver.source_term, solver.buffer) for transform! in solver.transforms.forward]

    # Solve tridiagonal system of linear equations in z at every column.
    solve!(ϕ, solver.batched_tridiagonal_solver, solver.source_term)

    # Apply backward transforms in order
    [transform!(ϕ, solver.buffer) for transform! in solver.transforms.backward]

    ϕ .= real.(ϕ)

    # Set the volume mean of the solution to be zero.
    # Solutions to Poisson's equation are only unique up to a constant (the global mean
    # of the solution), so we need to pick a constant. We choose the constant to be zero
    # so that the solution has zero-mean.
    ϕ .= ϕ .- mean(ϕ)

    @apply_regionally redistribute_real_component!(x, ϕ, arch, grid, Iterate(1:length(grid)), grid.partition)

    return nothing
end

####
#### Redistribute real component
####

function redistribute_real_component!(ϕ, ϕc, arch, grid, region, partition)
    copy_event = launch!(arch, grid, :xyz, _redistribute_real_component!, ϕ, ϕc, grid, region, partition, dependencies=device_event(arch))
    wait(device(arch), copy_event)
end

@kernel function _redistribute_real_component!(ϕ, ϕc, grid, region, partition)
    i, j, k = @index(Global, NTuple)

    i′, j′, k′ = global_index(i, j, k, grid, region, partition)
    @inbounds ϕ[i, j, k] = real(ϕc[i′, j′, k′])
end