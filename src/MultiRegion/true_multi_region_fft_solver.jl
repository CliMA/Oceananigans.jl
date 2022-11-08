using Oceananigans.Architectures: device_event
using FFTW
using CUDA, CUDA.CUFFT
using Oceananigans.Operators

using Oceananigans.Grids 
using Oceananigans.Fields: indices
using Oceananigans.Solvers: poisson_eigenvalues, plan_transforms, copy_real_component!
using Oceananigans.Solvers: plan_forward_transform, plan_backward_transform, Backward, Forward, DiscreteTransform
using Oceananigans.Models.NonhydrostaticModels: calculate_pressure_source_term_fft_based_solver!
using Oceananigans.Models.NonhydrostaticModels: calculate_pressure_source_term_fourier_tridiagonal_solver!

import Oceananigans.Architectures: architecture
import Oceananigans.Models.NonhydrostaticModels: PressureSolver, solve_for_pressure!
import Oceananigans.Solvers: solve!

struct MultiRegionFFTBasedPoissonSolver{G, P, Λ, S, B, T}
           grid :: G
transposed_grid :: P
    eigenvalues :: Λ
        storage :: S
         buffer :: B
     operations :: T
end

# PressureSolver(arch, grid::RegMultiRegionGrid, planner_flag=FFTW.PATIENT) = MultiRegionFFTBasedPoissonSolver(grid, planner_flag)

function MultiRegionFFTBasedPoissonSolver(grid::MultiRegionGrid, planner_flag=FFTW.PATIENT)
    
    global_grid = reconstruct_global_grid(grid)
    tgrid = transposed_grid(grid)
    
    regional_arch = construct_regionally(architecture, grid)

    TX, TY, TZ = topo = topology(global_grid)

    arch = architecture(global_grid)
    storage = (initial    = construct_regionally(arch_array, regional_arch, zeros(complex(eltype(global_grid)), size(grid))), 
               transposed = construct_regionally(arch_array, regional_arch, zeros(complex(eltype(global_grid)), size(tgrid))))
    
    eigenvalues = construct_regional_eigenvalues(global_grid, tgrid)
    operations  = plan_multi_region_transforms(global_grid, grid, tgrid, storage, planner_flag)

    buffer_needed = arch isa GPU && Bounded in topo
    buffer = buffer_needed ? (initial    = construct_regionally(arch_array, regional_arch, zeros(complex(eltype(global_grid)), size(grid))), 
                              transposed = construct_regionally(arch_array, regional_arch, zeros(complex(eltype(global_grid)), size(tgrid)))) : 
                             (initial    = nothing, 
                              transposed = nothing)

    return MultiRegionFFTBasedPoissonSolver(grid, tgrid, eigenvalues, storage, buffer, operations)
end

function construct_regional_eigenvalues(global_grid, tgrid) 
    
    TX, TY, TZ = topology(global_grid)

    λx = poisson_eigenvalues(global_grid.Nx, global_grid.Lx, 1, TX())
    λy = poisson_eigenvalues(global_grid.Ny, global_grid.Ly, 2, TY())
    λz = poisson_eigenvalues(global_grid.Nz, global_grid.Lz, 3, TZ())

    arch = architecture(global_grid)

    p       = tgrid.partition
    regions = Iterate(1:length(p))
    tsize   = construct_regionally(size, tgrid)
 
    return distribute_regional_eigenvalues(λx, λy, λz, p, regions, tsize, arch)
end

distribute_regional_eigenvalues(λx, λy, λz, p::XPartition, regions, tsize, arch) = 
                  (λx = construct_regionally(partition_global_array, λx, p, tsize, regions, arch),
                   λy = construct_regionally(arch_array, arch, λy),
                   λz = construct_regionally(arch_array, arch, λz))

distribute_regional_eigenvalues(λx, λy, λz, p::YPartition, regions, tsize, arch) = 
                  (λx = construct_regionally(arch_array, arch, λx),
                   λy = construct_regionally(partition_global_array, λy, p, tsize, regions, arch),
                   λz = construct_regionally(arch_array, arch, λz))

function solve_for_pressure!(pressure, solver::MultiRegionFFTBasedPoissonSolver, Δt, U★)

    # Calculate right hand side:
    arch = architecture(solver.grid)
    rhs  = solver.storage.initial
    grid = solver.grid

    @apply_regionally calculate_source_term!(rhs, arch, grid, Δt, U★, solver)

    # Solve pressure Poisson given for pressure, given rhs
    solve!(pressure, solver)

    return nothing
end

function calculate_source_term!(rhs, arch, grid, Δt, U★, ::MultiRegionFFTBasedPoissonSolver)
    rhs_event = launch!(arch, grid, :xyz, calculate_pressure_source_term_fft_based_solver!,
                        rhs, grid, Δt, U★, dependencies = device_event(arch))
    wait(device(arch), rhs_event)
end

# function calculate_source_term!(rhs, arch, grid, Δt, U★, ::MultiRegionFourierTridiagonalSolver)
#     rhs_event = launch!(arch, grid, :xyz, calculate_pressure_source_term_fourier_tridiagonal_solver!,
#                         rhs, grid, Δt, U★, dependencies = device_event(arch))
#     wait(device(arch), rhs_event)
# end

function copy_event!(ϕ, ϕc, arch, grid)
    copy_event = launch!(arch, grid, :xyz, copy_real_component!, ϕ, ϕc, indices(ϕ), dependencies=device_event(arch))
    wait(device(arch), copy_event)
end

@inline apply_multi_operation(func, args...) = func(args...)

function apply_parallel_transforms(transform1!, transform2!, storage, buffer)
    transform1!(storage, buffer)
    transform2!(storage, buffer)
end

@inline function divide_by_eigenvalues!(ϕc, b, λx, λy, λz, m, arch, grid) 
    divide_event = launch!(arch, grid, :xyz, _divide_by_eigenvalues!, ϕc, b, λx, λy, λz, m, dependencies=device_event(arch))
    wait(device(arch), divide_event)
end

@kernel function _divide_by_eigenvalues!(ϕc, b, λx, λy, λz, m)
    i, j, k = @index(Global, NTuple)
    ϕc[i, j, k] = - b[i, j, k] / (λx[i, 1, 1] + λy[1, j, 1] + λz[1, 1, k] - m)
end

@inline function set_first_element_to_zero!(storage, region, m)
    if region == 1 && m === 0
        CUDA.@allowscalar storage[1, 1, 1] = 0
    end
end

function solve!(ϕ, solver::MultiRegionFFTBasedPoissonSolver, m=0)
    
    arch    = architecture(solver.grid)
    grid    = solver.grid
    tgrid   = solver.transposed_grid
    storage = solver.storage
    buffer  = solver.buffer
    
    operations = solver.operations
    λx, λy, λz = solver.eigenvalues

    # Transform b *in-place* to eigenfunction space
    @apply_regionally begin
        apply_parallel_transforms(operations.forward[1], operations.forward[2], storage.initial, buffer.initial)
    end
    operations.forward[3](storage.transposed, storage.initial, tgrid, grid)
    @apply_regionally apply_multi_operation(operations.forward[4], storage.transposed, buffer.transposed)

    # Solve the discrete screened Poisson equation (∇² + m) ϕ = b.
    @apply_regionally begin
        divide_by_eigenvalues!(storage.transposed, storage.transposed, λx, λy, λz, m, arch, tgrid)
        set_first_element_to_zero!(storage.transposed, Iterate(1:length(grid.partition)), m)
    end

    # Apply backward transforms in order
    @apply_regionally apply_multi_operation(operations.backward[1], storage.transposed, buffer.transposed)
    operations.backward[2](storage.initial, storage.transposed, grid, tgrid)
    
    @apply_regionally begin
        apply_parallel_transforms(operations.backward[3], operations.backward[4], storage.initial, buffer.initial)
        copy_event!(ϕ, storage.initial, arch, grid)
    end

    return ϕ
end
