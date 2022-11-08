using Oceananigans.Architectures: device_event
using FFTW
using Statistics: mean
using CUDA, CUDA.CUFFT
using Oceananigans.Operators: Δzᵃᵃᶠ

using Oceananigans.Grids 
using Oceananigans.Operators
using Oceananigans.Fields: indices
using Oceananigans.Solvers: poisson_eigenvalues, plan_transforms, copy_real_component!, compute_main_diagonals!
using Oceananigans.Solvers: plan_forward_transform, plan_backward_transform, Backward, Forward, DiscreteTransform
using Oceananigans.Models.NonhydrostaticModels: calculate_pressure_source_term_fft_based_solver!
using Oceananigans.Models.NonhydrostaticModels: calculate_pressure_source_term_fourier_tridiagonal_solver!

import Oceananigans.Architectures: architecture
import Oceananigans.Models.NonhydrostaticModels: PressureSolver, solve_for_pressure!
import Oceananigans.Solvers: solve!

struct MultiRegionFourierTridiagonalSolver{G, P, B, S, β, T, R}
                      grid :: G
           transposed_grid :: P
batched_tridiagonal_solver :: B
                   storage :: S
                    buffer :: β 
                operations :: T
               source_term :: R
end

PressureSolver(arch, grid::HRegMultiRegionGrid, planner_flag=FFTW.PATIENT) = MultiRegionFourierTridiagonalSolver(grid, planner_flag)

function MultiRegionFourierTridiagonalSolver(grid::MultiRegionGrid, planner_flag=FFTW.PATIENT)
    
    global_grid = reconstruct_global_grid(grid)
    tgrid = transposed_grid(grid)
    
    regional_arch = construct_regionally(architecture, grid)

    arch = architecture(global_grid)
    storage = (initial    = construct_regionally(arch_array, regional_arch, zeros(complex(eltype(global_grid)), size(grid))), 
               transposed = construct_regionally(arch_array, regional_arch, zeros(complex(eltype(global_grid)), size(tgrid))))
    
    source_term = (initial    = construct_regionally(arch_array, regional_arch, zeros(complex(eltype(global_grid)), size(grid))), 
                   transposed = construct_regionally(arch_array, regional_arch, zeros(complex(eltype(global_grid)), size(tgrid))))

    TX, TY, TZ = topology(global_grid)
    TZ != Bounded && error("FourierTridiagonalPoissonSolver can only be used with a Bounded z topology.")

    arch = architecture(global_grid)

    operations  = plan_multi_region_transforms(global_grid, grid, tgrid, storage, planner_flag)

    buffer = (initial    = construct_regionally(arch_array, regional_arch, zeros(complex(eltype(global_grid)), size(grid))), 
              transposed = construct_regionally(arch_array, regional_arch, zeros(complex(eltype(global_grid)), size(tgrid)))) 
              
    # Lower and upper diagonals are the same
    lower_diagonal = CUDA.@allowscalar [1 / Δzᵃᵃᶠ(1, 1, k, global_grid) for k in 2:size(global_grid, 3)]
    lower_diagonal = construct_regionally(arch_array, regional_arch, lower_diagonal)
    upper_diagonal = lower_diagonal

    eigenvalues = construct_regional_eigenvalues(global_grid, tgrid)

    # Compute diagonal coefficients for each grid point
    diagonal = construct_regionally(build_main_diagonals, regional_arch, tgrid, eigenvalues.λx, eigenvalues.λy)

    # Set up batched tridiagonal solver
    btsolver = construct_regionally(BatchedTridiagonalSolver, tgrid;
                                        lower_diagonal = lower_diagonal,
                                              diagonal = diagonal,
                                        upper_diagonal = upper_diagonal)

    return MultiRegionFourierTridiagonalSolver(grid, tgrid, btsolver, storage, buffer, operations, source_term)
end

function build_main_diagonals(arch, grid, λx, λy)
    diagonal = arch_array(arch, zeros(size(grid)))

    event = launch!(arch, grid, :xy, compute_main_diagonals!, diagonal, grid, λx, λy, dependencies=device_event(arch))
    wait(device(arch), event)

    return diagonal
end

function solve_for_pressure!(pressure, solver::MultiRegionFourierTridiagonalSolver, Δt, U★)

    # Calculate right hand side:
    arch = architecture(solver.grid)
    rhs  = solver.source_term.initial
    grid = solver.grid

    @apply_regionally calculate_source_term!(rhs, arch, grid, Δt, U★, solver)

    # Solve pressure Poisson given for pressure, given rhs
    solve!(pressure, solver)

    return nothing
end

function calculate_source_term!(rhs, arch, grid, Δt, U★, ::MultiRegionFourierTridiagonalSolver)
    rhs_event = launch!(arch, grid, :xyz, calculate_pressure_source_term_fourier_tridiagonal_solver!,
                        rhs, grid, Δt, U★, dependencies = device_event(arch))
    wait(device(arch), rhs_event)
end

function solve!(ϕ, solver::MultiRegionFourierTridiagonalSolver, m=0)
    
    arch    = architecture(solver.grid)
    grid    = solver.grid
    tgrid   = solver.transposed_grid
    storage = solver.storage
    buffer  = solver.buffer
 
    source_term = solver.source_term   
    operations  = solver.operations
    btsolver    = solver.batched_tridiagonal_solver

    # Transform b *in-place* to eigenfunction space
    @apply_regionally apply_multi_operation(operations.forward[1], source_term.initial, buffer.initial)
    operations.forward[2](source_term.transposed, source_term.initial, tgrid, grid)
    @apply_regionally apply_multi_operation(operations.forward[3], source_term.transposed, buffer.transposed)

    # Solve the discrete screened Poisson equation (∇² + m) ϕ = b.
    @apply_regionally solve!(storage.transposed, btsolver, source_term.transposed)

    # Apply backward transforms in order
    @apply_regionally apply_multi_operation(operations.backward[1], storage.transposed, buffer.transposed)
    operations.backward[2](storage.initial, storage.transposed, grid, tgrid)
    @apply_regionally apply_multi_operation(operations.backward[3], storage.initial, buffer.initial)
    
    mean_val = construct_regionally(mean, storage.initial)
    mean_val = mean(mean_val.regions)

    @apply_regionally copy_event_tridiagonal!(ϕ, storage.initial, mean_val, arch, grid)
    
    return ϕ
end

function copy_event_tridiagonal!(ϕ, ϕc, mean_val, arch, grid)
    copy_event = launch!(arch, grid, :xyz, copy_real_component!, ϕ, ϕc, indices(ϕ), dependencies=device_event(arch))
    wait(device(arch), copy_event)

    ϕ .-= real(mean_val)
end
