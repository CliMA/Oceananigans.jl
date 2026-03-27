using Oceananigans.Architectures: array_type
using Oceananigans.Solvers: plan_forward_transform, plan_backward_transform, DiscreteTransform
using Oceananigans.Solvers: Forward, Backward

@inline reshaped_size(grid) = size(grid, 2), size(grid, 1), size(grid, 3)

function plan_distributed_transforms(global_grid, storage::TransposableField, planner_flag)
    topo = topology(global_grid)
    arch = architecture(global_grid)

    grids = (storage.zfield.grid, storage.yfield.grid, storage.xfield.grid)

    forward_plan_x  =  plan_forward_transform(parent(storage.xfield), topo[1](), [1], planner_flag)
    forward_plan_z  =  plan_forward_transform(parent(storage.zfield), topo[3](), [3], planner_flag)
    backward_plan_x = plan_backward_transform(parent(storage.xfield), topo[1](), [1], planner_flag)
    backward_plan_z = plan_backward_transform(parent(storage.zfield), topo[3](), [3], planner_flag)

    # For Periodic y-topology on GPU, plan the y-FFT along dim 2 directly.
    # This avoids the costly permutedims operations that the old reshape+dim1 approach required.
    # For Bounded y-topology on GPU, we still use the reshape approach because the
    # twiddle factors and index permutations assume dim-1 layout.
    if arch isa GPU && topo[2] == Bounded
        rs_size    = reshaped_size(grids[2])
        rs_storage = reshape(parent(storage.yfield), rs_size)
        forward_plan_y  =  plan_forward_transform(rs_storage, topo[2](), [1], planner_flag)
        backward_plan_y = plan_backward_transform(rs_storage, topo[2](), [1], planner_flag)
        y_dims = [2]  # DiscreteTransform dims — triggers transpose_dims=(2,1,3) for Bounded
    else
        # Periodic GPU and all CPU: plan along dim 2 directly (no reshape needed)
        forward_plan_y  =  plan_forward_transform(parent(storage.yfield), topo[2](), [2], planner_flag)
        backward_plan_y = plan_backward_transform(parent(storage.yfield), topo[2](), [2], planner_flag)
        y_dims = [2]  # For Periodic, DiscreteTransform will set transpose_dims=nothing
    end

    forward_operations = (
        z! = DiscreteTransform(forward_plan_z, Forward(), grids[1], [3]),
        y! = DiscreteTransform(forward_plan_y, Forward(), grids[2], y_dims),
        x! = DiscreteTransform(forward_plan_x, Forward(), grids[3], [1]),
    )

    backward_operations = (
        x! = DiscreteTransform(backward_plan_x, Backward(), grids[3], [1]),
        y! = DiscreteTransform(backward_plan_y, Backward(), grids[2], y_dims),
        z! = DiscreteTransform(backward_plan_z, Backward(), grids[1], [3]),
    )

    return (; forward = forward_operations, backward = backward_operations)
end
