using Oceananigans.Architectures: array_type
using Oceananigans.Solvers: plan_forward_transform, plan_backward_transform, DiscreteTransform
using Oceananigans.Solvers: Forward, Backward

@inline reshaped_size(grid) = size(grid, 2), size(grid, 1), size(grid, 3)

struct FFTPlan{F, B}
    forward :: F
    backward :: B
end

function plan_distributed_transforms(global_grid, storage::ParallelFields, planner_flag)
    topo = topology(global_grid)
    arch = architecture(global_grid)

    grids = (storage.zfield.grid, storage.yfield.grid, storage.xfield.grid)

    rs_size    = reshaped_size(grids[2])
    rs_storage = reshape(parent(storage.yfield), rs_size)

    forward_plan_x  =  plan_forward_transform(parent(storage.xfield), topo[1](), [1], planner_flag)
    forward_plan_z  =  plan_forward_transform(parent(storage.zfield), topo[3](), [3], planner_flag)
    backward_plan_x = plan_backward_transform(parent(storage.xfield), topo[1](), [1], planner_flag)
    backward_plan_z = plan_backward_transform(parent(storage.zfield), topo[3](), [3], planner_flag)

    if arch isa GPU
        forward_plan_y  =  plan_forward_transform(rs_storage, topo[2](), [1], planner_flag) 
        backward_plan_y = plan_backward_transform(rs_storage, topo[2](), [1], planner_flag) 
    else
        forward_plan_y  =  plan_forward_transform(parent(storage.yfield), topo[2](), [2], planner_flag) 
        backward_plan_y = plan_backward_transform(parent(storage.yfield), topo[2](), [2], planner_flag) 
    end

    forward_operations = (
        z! = DiscreteTransform(forward_plan_z, Forward(), grids[1], [3]),
        y! = DiscreteTransform(forward_plan_y, Forward(), grids[2], [2]),
        x! = DiscreteTransform(forward_plan_x, Forward(), grids[3], [1]),
    )

    backward_operations = (
        x! = DiscreteTransform(backward_plan_x, Backward(), grids[3], [1]),
        y! = DiscreteTransform(backward_plan_y, Backward(), grids[2], [2]),
        z! = DiscreteTransform(backward_plan_z, Backward(), grids[1], [3]),
    )

    return FFTPlan(forward_operations, backward_operations)
end