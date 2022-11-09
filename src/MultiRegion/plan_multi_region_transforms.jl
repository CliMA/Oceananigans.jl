forward_orders(::Type{Periodic}, ::Type{Bounded})  = (1, 2)
forward_orders(::Type{Bounded},  ::Type{Periodic}) = (2, 1)
forward_orders(::Type{Bounded},  ::Type{Bounded})  = (1, 2)
forward_orders(::Type{Periodic}, ::Type{Periodic}) = (1, 2)

backward_orders(::Type{Periodic}, ::Type{Bounded})  = (2, 1)
backward_orders(::Type{Bounded},  ::Type{Periodic}) = (1, 2)
backward_orders(::Type{Bounded},  ::Type{Bounded})  = (1, 2)
backward_orders(::Type{Periodic}, ::Type{Periodic}) = (1, 2)

@inline reshaped_size(grid) = size(grid, 2), size(grid, 1), size(grid, 3)

function plan_multi_region_transforms(global_grid::RegRectilinearGrid, grid::XPartitionedGrid, transposed_grid::YPartitionedGrid, storage, planner_flag)
    TX, TY, TZ = topo = topology(global_grid)
    arch = architecture(global_grid)

    N = construct_regionally(reshaped_size, grid)
    # Convert Flat to Bounded for inferring batchability and transform ordering
    # Note that transforms are omitted in Flat directions.
    unflattened_topo = Tuple(T() isa Flat ? Bounded : T for T in (TY, TZ))

    rs_storage = construct_regionally(reshape, storage.initial, N)

    forward_plan_x  = construct_regionally(plan_forward_transform,  storage.transposed, topo[1](), [1], planner_flag)
    forward_plan_z  = construct_regionally(plan_forward_transform,  storage.initial,    topo[3](), [3], planner_flag)
    backward_plan_x = construct_regionally(plan_backward_transform, storage.transposed, topo[1](), [1], planner_flag)
    backward_plan_z = construct_regionally(plan_backward_transform, storage.initial,    topo[3](), [3], planner_flag)

    if arch isa GPU
        forward_plan_y  = construct_regionally(plan_forward_transform,  rs_storage, topo[2](), [1], planner_flag) 
        backward_plan_y = construct_regionally(plan_backward_transform, rs_storage, topo[2](), [1], planner_flag) 
    else
        forward_plan_y  = construct_regionally(plan_forward_transform,  storage.initial, topo[2](), [2], planner_flag) 
        backward_plan_y = construct_regionally(plan_backward_transform, storage.initial, topo[2](), [2], planner_flag) 
    end

    forward_plans  = (forward_plan_y,  forward_plan_z)
    backward_plans = (backward_plan_y, backward_plan_z)

    f_order = forward_orders(unflattened_topo...)
    b_order = backward_orders(unflattened_topo...)

    forward_operations = (
        construct_regionally(DiscreteTransform, forward_plans[f_order[1]], Forward(), grid, [[2, 3][f_order[1]]]),
        construct_regionally(DiscreteTransform, forward_plans[f_order[2]], Forward(), grid, [[2, 3][f_order[2]]]),
        transpose_x_to_y!,
        construct_regionally(DiscreteTransform, forward_plan_x, Forward(), transposed_grid, [1]),
    )

    backward_operations = (
        construct_regionally(DiscreteTransform, backward_plan_x, Backward(), transposed_grid, [1]),
        transpose_y_to_x!,
        construct_regionally(DiscreteTransform, backward_plans[b_order[1]], Backward(), grid, [[2, 3][b_order[1]]]),
        construct_regionally(DiscreteTransform, backward_plans[b_order[2]], Backward(), grid, [[2, 3][b_order[2]]]),
    )

    return (; forward = forward_operations, backward = backward_operations)
end

function plan_multi_region_transforms(global_grid::RegRectilinearGrid, grid::YPartitionedGrid, transposed_grid::XPartitionedGrid, storage, planner_flag)
    TX, TY, TZ = topo = topology(global_grid)
    arch = architecture(global_grid)

    N = construct_regionally(reshaped_size, transposed_grid)
    # Convert Flat to Bounded for inferring batchability and transform ordering
    # Note that transforms are omitted in Flat directions.
    unflattened_topo = Tuple(T() isa Flat ? Bounded : T for T in (TX, TZ))

    rs_storage = construct_regionally(reshape, storage.transposed, N)

    forward_plan_x = construct_regionally(plan_forward_transform, storage.initial, topo[1](), [1], planner_flag)
    forward_plan_z = construct_regionally(plan_forward_transform, storage.initial, topo[3](), [3], planner_flag)

    backward_plan_x = construct_regionally(plan_backward_transform, storage.initial, topo[1](), [1], planner_flag)
    backward_plan_z = construct_regionally(plan_backward_transform, storage.initial, topo[3](), [3], planner_flag)

    if arch isa GPU
        forward_plan_y  = construct_regionally(plan_forward_transform,  rs_storage, topo[2](), [1], planner_flag) 
        backward_plan_y = construct_regionally(plan_backward_transform, rs_storage, topo[2](), [1], planner_flag) 
    else
        forward_plan_y  = construct_regionally(plan_forward_transform,  storage.transposed, topo[2](), [2], planner_flag) 
        backward_plan_y = construct_regionally(plan_backward_transform, storage.transposed, topo[2](), [2], planner_flag) 
    end

    forward_plans  = (forward_plan_x,  forward_plan_z)
    backward_plans = (backward_plan_x, backward_plan_z)

    f_order = forward_orders(unflattened_topo...)
    b_order = backward_orders(unflattened_topo...)

    forward_operations = (
        construct_regionally(DiscreteTransform, forward_plans[f_order[1]], Forward(), grid, [[1, 3][f_order[1]]]),
        construct_regionally(DiscreteTransform, forward_plans[f_order[2]], Forward(), grid, [[1, 3][f_order[2]]]),
        transpose_y_to_x!,
        construct_regionally(DiscreteTransform, forward_plan_y, Forward(), transposed_grid, [2]),
    )

    backward_operations = (
        construct_regionally(DiscreteTransform, backward_plan_y, Backward(), transposed_grid, [2]),
        transpose_x_to_y!,
        construct_regionally(DiscreteTransform, backward_plans[b_order[1]], Backward(), grid, [[1, 3][b_order[1]]]),
        construct_regionally(DiscreteTransform, backward_plans[b_order[2]], Backward(), grid, [[1, 3][b_order[2]]]),
    )

    return (; forward = forward_operations, backward = backward_operations)
end

""" Used by MultiRegionFourierTridiagonalPoissonSolver. """
function plan_multi_region_transforms(global_grid::HRegRectilinearGrid, grid::XPartitionedGrid, transposed_grid::YPartitionedGrid, storage, planner_flag)
    topo = topology(global_grid)
    arch = architecture(global_grid)

    N = construct_regionally(reshaped_size, grid)

    rs_storage = construct_regionally(reshape, storage.initial, N)

    forward_plan_x  = construct_regionally(plan_forward_transform,  storage.transposed, topo[1](), [1], planner_flag)
    backward_plan_x = construct_regionally(plan_backward_transform, storage.transposed, topo[1](), [1], planner_flag)

    if arch isa GPU
        forward_plan_y  = construct_regionally(plan_forward_transform,  rs_storage, topo[2](), [1], planner_flag) 
        backward_plan_y = construct_regionally(plan_backward_transform, rs_storage, topo[2](), [1], planner_flag) 
    else
        forward_plan_y  = construct_regionally(plan_forward_transform,  storage.initial, topo[2](), [2], planner_flag) 
        backward_plan_y = construct_regionally(plan_backward_transform, storage.initial, topo[2](), [2], planner_flag) 
    end

    forward_operations = (
        construct_regionally(DiscreteTransform, forward_plan_y, Forward(), grid, [2]),
        transpose_x_to_y!,
        construct_regionally(DiscreteTransform, forward_plan_x, Forward(), transposed_grid, [1]),
    )

    backward_operations = (
        construct_regionally(DiscreteTransform, backward_plan_x, Backward(), transposed_grid, [1]),
        transpose_y_to_x!,
        construct_regionally(DiscreteTransform, backward_plan_y, Backward(), grid, [2]),
    )

    return (; forward = forward_operations, backward = backward_operations)
end

function plan_multi_region_transforms(global_grid::HRegRectilinearGrid, grid::YPartitionedGrid, transposed_grid::XPartitionedGrid, storage, planner_flag)
    topo = topology(global_grid)
    arch = architecture(global_grid)

    N = construct_regionally(reshaped_size, transposed_grid)

    rs_storage = construct_regionally(reshape, storage.transposed, N)

    forward_plan_x  = construct_regionally(plan_forward_transform,  storage.initial, topo[1](), [1], planner_flag)
    backward_plan_x = construct_regionally(plan_backward_transform, storage.initial, topo[1](), [1], planner_flag)

    if arch isa GPU
        forward_plan_y  = construct_regionally(plan_forward_transform,  rs_storage, topo[2](), [1], planner_flag) 
        backward_plan_y = construct_regionally(plan_backward_transform, rs_storage, topo[2](), [1], planner_flag) 
    else
        forward_plan_y  = construct_regionally(plan_forward_transform,  storage.transposed, topo[2](), [2], planner_flag) 
        backward_plan_y = construct_regionally(plan_backward_transform, storage.transposed, topo[2](), [2], planner_flag) 
    end

    forward_operations = (
        construct_regionally(DiscreteTransform, forward_plan_x, Forward(), grid, [1]),
        transpose_y_to_x!,
        construct_regionally(DiscreteTransform, forward_plan_y, Forward(), transposed_grid, [2]),
    )

    backward_operations = (
        construct_regionally(DiscreteTransform, backward_plan_y, Backward(), transposed_grid, [2]),
        transpose_x_to_y!,
        construct_regionally(DiscreteTransform, backward_plan_x, Backward(), grid, [1]),
    )

    return (; forward = forward_operations, backward = backward_operations)
end