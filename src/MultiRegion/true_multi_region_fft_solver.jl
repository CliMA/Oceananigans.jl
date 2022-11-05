using Oceananigans.Architectures: device_event
using FFTW
using CUDA, CUDA.CUFFT
using Oceananigans.Operators

import Oceananigans.Architectures: architecture
import Oceananigans.Models.NonhydrostaticModels: PressureSolver, solve_for_pressure!
import Oceananigans.Solvers: FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver, solve!

using Oceananigans.Grids 
using Oceananigans.Solvers: poisson_eigenvalues, plan_transforms, copy_real_component!

struct MultiRegionFFTBasedPoissonSolver{G, P, Λ, S, B, T}
           grid :: G
transposed_grid :: P
    eigenvalues :: Λ
        storage :: S
     operations :: T
end

PressureSolver(arch, grid::RegMultiRegionGrid)  = MultiRegionFFTBasedPoissonSolver(grid)
PressureSolver(arch, grid::HRegMultiRegionGrid) = MultiRegionPoissonSolver(grid, FourierTridiagonalPoissonSolver(grid))

function MultiRegionFFTBasedPoissonSolver(grid::MultiRegionGrid, planner_flag=FFTW.PATIENT)
    
    global_grid     = reconstruct_global_grid(grid)

    transposed_grid = transpose_grid(global_grid, grid.partition, grid.devices)

    arch = architecture(global_grid)
    storage = (initial    = construct_regionally(arch_array, arch, zeros(complex(eltype(global_grid)), size(grid))), 
               transposed = construct_regionally(arch_array, arch, zeros(complex(eltype(global_grid)), size(transposed_grid))))
    
    TX, TY, TZ = topology(global_grid)

    λx = poisson_eigenvalues(grid.Nx, grid.Lx, 1, TX())
    λy = poisson_eigenvalues(grid.Ny, grid.Ly, 2, TY())
    λz = poisson_eigenvalues(grid.Nz, grid.Lz, 3, TZ())

    arch = architecture(global_grid)

    p       = transpose_grid.partition
    regions = Iterate(1:length(p))
    tsize   = construct_regionally(size, transposed_grid)

    eigenvalues = (λx = construct_regionally(partition_global_array, λx, p, tsize, regions, arch),
                   λy = construct_regionally(partition_global_array, λy, p, tsize, regions, arch),
                   λz = construct_regionally(partition_global_array, λz, p, tsize, regions, arch))

    transforms = plan_multi_region_transforms(global_grid, grid, transposed_grid, storage, planner_flag)
    
    return MultiRegionFFTBasedPoissonSolver(grid, transposed_grid, eigenvalues, storage, transforms)
end

forward_orders(::Type{Periodic}, ::Type{Bounded}) = (1, 2)
forward_orders(::Type{Bounded}, ::Type{Periodic}) = (2, 1)

backward_orders(::Type{Periodic}, ::Type{Bounded}) = (2, 1)
backward_orders(::Type{Bounded}, ::Type{Periodic}) = (1, 2)

function plan_multi_region_transforms(global_grid, grid::XPartitionedGrid, transposed_grid, storage, planner_flag)
    Nx, Ny, Nz = size(global_grid)
    topo = topology(global_grid)

    periodic_dims = findall(t -> t == Periodic, topo)
    bounded_dims  = findall(t -> t == Bounded, topo)

    # Convert Flat to Bounded for inferring batchability and transform ordering
    # Note that transforms are omitted in Flat directions.
    unflattened_topo = Tuple(T() isa Flat ? Bounded : T for T in topo[2:3])

    arch = architecture(grid)

    forward_plan_x = construct_regionally(plan_forward_transform, storage.transposed, topo[1](), [1], planner_flag)
    forward_plan_y = construct_regionally(plan_forward_transform, storage.initial,    topo[2](), [2], planner_flag)
    forward_plan_z = construct_regionally(plan_forward_transform, storage.initial,    topo[3](), [3], planner_flag)

    backward_plan_x = construct_regionally(plan_backward_transform, storage.transposed, topo[1](), [1], planner_flag)
    backward_plan_y = construct_regionally(plan_backward_transform, storage.initial,    topo[2](), [2], planner_flag)
    backward_plan_z = construct_regionally(plan_backward_transform, storage.initial,    topo[3](), [3], planner_flag)

    forward_plans  = (forward_plan_x, forward_plan_y,  forward_plan_z)
    backward_plans = (backward_plan_x, backward_plan_y, backward_plan_z)

    f_order = forward_orders(unflattened_topo...)
    b_order = backward_orders(unflattened_topo...)

    forward_operations = (
        construct_regionally(DiscreteTransform, forward_plans[f_order[1]], Forward(), grid, [[2, 3][f_order[1]]]),
        construct_regionally(DiscreteTransform, forward_plans[f_order[2]], Forward(), grid, [[2, 3][f_order[2]]]),
        transpose_x_to_y!,
        construct_regionally(DiscreteTransform, forward_plan_x, Forward(), transposed_grid, [1]),
    )

    backward_transforms = (
        construct_regionally(DiscreteTransform, backward_plan_x, Backwards(), transposed_grid, [1]),
        transpose_y_to_x!,
        construct_regionally(DiscreteTransform, backward_plans[b_order[1]], Backwards(), grid, [[2, 3][b_order[1]]]),
        construct_regionally(DiscreteTransform, backward_plans[b_order[2]], Backwards(), grid, [[2, 3][f_order[2]]]),
    )
end

transform_or_transpose!(operation!, storage, b)                    = operation!(storage.initial, storage.transformed)
transform_or_transpose!(operation!::DiscreteTransform, storage, b) = @apply_regionally operation!(b)

using Oceananigans.Models.NonhydrostaticModels: calculate_pressure_source_term_fft_based_solver!, copy_real_component!

function solve_for_pressure!(pressure, solver::MultiRegionFFTBasedPoissonSolver, Δt, U★)

    # Calculate right hand side:
    rhs  = solver.storage.initial
    arch = architecture(solver.grid)
    grid = solver.grid

    regions = Iterate(1:length(grid))

    @apply_regionally calculate_pressure_source_term_fft_based_solver!(rhs, grid, Δt, U★)

    # Solve pressure Poisson given for pressure, given rhs
    solve!(pressure, solver, rhs)

    return nothing
end

function copy_event!(ϕ, ϕc, arch, grid)
    copy_event = launch!(arch, grid, :xyz, copy_real_component!, ϕ, ϕc, indices(ϕ), dependencies=device_event(arch))
    wait(device(arch), copy_event)
end

function solve!(ϕ, solver::MultiRegionFFTBasedPoissonSolver, b, m=0)
    
    arch   = architecture(solver)
    grid   = multi_solver.grid

    λx, λy, λz = solver.eigenvalues

    # Temporarily store the solution in ϕc
    ϕc = solver.storage.transposed

    # Transform b *in-place* to eigenfunction space
    [transform_or_transpose!(b) for transform_or_transpose! in solver.operations.forward]

    # Solve the discrete screened Poisson equation (∇² + m) ϕ = b.
    @apply_regionally divide_by_eigenvalues!(ϕc, b, λx, λy, λz, m)

    # If m === 0, the "zeroth mode" at `i, j, k = 1, 1, 1` is undetermined;
    # we set this to zero by default. Another slant on this "problem" is that
    # λx[1, 1, 1] + λy[1, 1, 1] + λz[1, 1, 1] = 0, which yields ϕ[1, 1, 1] = Inf or NaN.
    m === 0 && CUDA.@allowscalar ϕc[1, 1, 1] = 0

    # Apply backward transforms in order
    [transform_or_transpose!(ϕc) for transform_or_transpose! in solver.operations.forward]

    [transform!(ϕc, solver.buffer) for transform! in solver.transforms.backward]

    @apply_regionally copy_event!(ϕ, ϕc, arch, grid)

    return ϕ
end

@inline divide_by_eigenvalues!(ϕc, b, λx, λy, λz, m) = @. ϕc = - b / (λx + λy + λz - m)