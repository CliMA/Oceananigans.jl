
# solve! requires that `b` in `A x = b` (the right hand side) 
# is copied in the solver storage
# See: Models/NonhydrostaticModels/solve_for_pressure.jl
function solve!(x, solver::DistributedFourierTridiagonalPoissonSolver{<:XYRegularRG})
    storage = solver.storage
    buffer  = solver.buffer

    transpose_z_to_y!(storage) # copy data from storage.zfield to storage.yfield
    solver.plan.forward.y!(parent(storage.yfield), buffer.y) 
    transpose_y_to_x!(storage) # copy data from storage.yfield to storage.xfield
    solver.plan.forward.x!(parent(storage.xfield), buffer.x)
    transpose_x_to_y!(storage) # copy data from storage.xfield to storage.yfield
    transpose_y_to_z!(storage) # copy data from storage.yfield to storage.zfield
  
    # Perform the implicit vertical solve here on storage.zfield...
    # Solve tridiagonal system of linear equations at every column.
    solve!(storage.zfield, solver.batched_tridiagonal_solver, solver.source_term)

    transpose_z_to_y!(storage)
    solver.plan.backward.y!(parent(storage.yfield), buffer.y)
    transpose_y_to_x!(storage) # copy data from storage.yfield to storage.xfield
    solver.plan.backward.y!(parent(storage.xfield), buffer.x)
    transpose_x_to_y!(storage) # copy data from storage.xfield to storage.yfield
    transpose_y_to_z!(storage) # copy data from storage.yfield to storage.zfield

    # Copy the real component of xc to x.
    launch!(arch, solver.local_grid, :xyz,
            _copy_real_component!, x, parent(storage.zfield))

    return x
end

function solve!(x, solver::DistributedFourierTridiagonalPoissonSolver{<:XZRegularRG})
    storage = solver.storage
    buffer  = solver.buffer

    solver.plan.forward.z!(parent(storage.zfield), buffer.z)
    transpose_z_to_y!(storage) # copy data from storage.zfield to storage.yfield
    transpose_y_to_x!(storage) # copy data from storage.yfield to storage.xfield
    solver.plan.forward.x!(parent(storage.xfield), buffer.x)
    transpose_x_to_y!(storage) # copy data from storage.xfield to storage.yfield
  
    # Perform the implicit vertical solve here on storage.zfield...
    # Solve tridiagonal system of linear equations at every column.
    solve!(storage.yfield, solver.batched_tridiagonal_solver, solver.source_term)

    transpose_y_to_x!(storage) # copy data from storage.yfield to storage.xfield
    solver.plan.backward.x!(parent(storage.xfield), buffer.x)
    transpose_x_to_y!(storage) # copy data from storage.xfield to storage.yfield
    transpose_y_to_z!(storage) # copy data from storage.yfield to storage.zfield
    solver.plan.backward.z!(parent(storage.zfield), buffer.z)

    # Copy the real component of xc to x.
    launch!(arch, solver.local_grid, :xyz,
            _copy_real_component!, x, parent(storage.zfield))

    return x
end

function solve!(x, solver::DistributedFourierTridiagonalPoissonSolver{<:YZRegularRG})
    storage = solver.storage
    buffer  = solver.buffer


    # Apply forward transforms to b = first(solver.storage).
    solver.plan.forward.z!(parent(storage.zfield), buffer.z)
    transpose_z_to_y!(storage) # copy data from storage.zfield to storage.yfield
    solver.plan.forward.y!(parent(storage.yfield), buffer.y) 
    transpose_y_to_x!(storage) # copy data from storage.yfield to storage.xfield

    # Perform the implicit vertical solve here on storage.zfield...
    # Solve tridiagonal system of linear equations at every column.
    solve!(storage.xfield, solver.batched_tridiagonal_solver, solver.source_term)

    transpose_x_to_y!(storage) # copy data from storage.xfield to storage.yfield
    solver.plan.backward.y!(parent(storage.yfield), buffer.y)
    transpose_y_to_z!(storage) # copy data from storage.yfield to storage.zfield
    solver.plan.backward.z!(parent(storage.zfield), buffer.z) # last backwards transform is in z

    # Copy the real component of xc to x.
    launch!(arch, solver.local_grid, :xyz,
            _copy_real_component!, x, parent(storage.zfield))

    return x
end