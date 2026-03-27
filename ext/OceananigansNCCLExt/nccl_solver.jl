using Oceananigans.Solvers: solve!, ZTridiagonalSolver, YTridiagonalSolver, XTridiagonalSolver

"""
    NCCLDistributedFFTSolver

Wraps a `DistributedFourierTridiagonalPoissonSolver` and replaces MPI-based
transposes with NCCL Send/Recv. The key advantage is eliminating `sync_device!`
calls before each transpose — NCCL operations are GPU-stream-native.

Construct from an existing solver:

    solver = DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid)
    nccl_solver = NCCLDistributedFFTSolver(solver)
"""
struct NCCLDistributedFFTSolver{S, XY, YZ}
    solver  :: S       # Inner DistributedFourierTridiagonalPoissonSolver
    nccl_xy :: XY      # NCCL comm for y↔x transpose (or nothing)
    nccl_yz :: YZ      # NCCL comm for z↔y transpose (or nothing)
end

function NCCLDistributedFFTSolver(solver::DC.DistributedFourierTridiagonalPoissonSolver)
    storage = solver.storage
    nccl_xy = storage.xybuff === nothing ? nothing : create_nccl_comm_from_mpi(storage.comms.xy)
    nccl_yz = storage.yzbuff === nothing ? nothing : create_nccl_comm_from_mpi(storage.comms.yz)
    return NCCLDistributedFFTSolver(solver, nccl_xy, nccl_yz)
end

# Forward architecture access
DC.architecture(s::NCCLDistributedFFTSolver) = DC.architecture(s.solver)

# Forward preconditioner RHS computation
Oceananigans.Solvers.compute_preconditioner_rhs!(s::NCCLDistributedFFTSolver, rhs) =
    Oceananigans.Solvers.compute_preconditioner_rhs!(s.solver, rhs)

# ─── Type aliases for dispatch ───

const NCCLZStretchedSolver = NCCLDistributedFFTSolver{
    <:DC.DistributedFourierTridiagonalPoissonSolver{<:Any, <:Any, <:ZTridiagonalSolver}}

const NCCLYStretchedSolver = NCCLDistributedFFTSolver{
    <:DC.DistributedFourierTridiagonalPoissonSolver{<:Any, <:Any, <:YTridiagonalSolver}}

const NCCLXStretchedSolver = NCCLDistributedFFTSolver{
    <:DC.DistributedFourierTridiagonalPoissonSolver{<:Any, <:Any, <:XTridiagonalSolver}}

# ─── Z-stretched solve (most common: slab-x with z-stretched grid) ───

function Oceananigans.Solvers.solve!(x, nccl_solver::NCCLZStretchedSolver)
    solver  = nccl_solver.solver
    storage = solver.storage

    if storage isa DC.SlabYFields
        return _nccl_slab_x_solve!(x, nccl_solver)
    else
        return _nccl_general_z_solve!(x, nccl_solver)
    end
end

# Optimized slab-x path: only 2 NCCL transposes (y↔x)
function _nccl_slab_x_solve!(x, nccl_solver::NCCLZStretchedSolver)
    solver  = nccl_solver.solver
    arch    = DC.architecture(solver)
    storage = solver.storage
    buffer  = solver.buffer

    # Forward: y-FFT (local) → NCCL transpose y→x → x-FFT
    solver.plan.forward.y!(parent(storage.yfield), buffer.y)
    nccl_transpose_y_to_x!(storage, nccl_solver.nccl_xy)
    solver.plan.forward.x!(parent(storage.xfield), buffer.x)

    # Tridiagonal solve in x-local space (z is fully local)
    parent(solver.source_term) .= parent(storage.xfield)
    solve!(storage.xfield, solver.batched_tridiagonal_solver, solver.source_term)

    # Backward: x-IFFT → NCCL transpose x→y → y-IFFT
    solver.plan.backward.x!(parent(storage.xfield), buffer.x)
    nccl_transpose_x_to_y!(storage, nccl_solver.nccl_xy)
    solver.plan.backward.y!(parent(storage.yfield), buffer.y)

    # Copy the real component (yfield aliases zfield for slab-x)
    launch!(arch, solver.local_grid, :xyz,
            DC._copy_real_component!, x, parent(storage.zfield))

    return x
end

# General Z-stretched solve (pencil decomposition): 4+ NCCL transposes
function _nccl_general_z_solve!(x, nccl_solver::NCCLZStretchedSolver)
    solver  = nccl_solver.solver
    arch    = DC.architecture(solver)
    storage = solver.storage
    buffer  = solver.buffer

    nccl_transpose_z_to_y!(storage, nccl_solver.nccl_yz)
    solver.plan.forward.y!(parent(storage.yfield), buffer.y)
    nccl_transpose_y_to_x!(storage, nccl_solver.nccl_xy)
    solver.plan.forward.x!(parent(storage.xfield), buffer.x)
    nccl_transpose_x_to_y!(storage, nccl_solver.nccl_xy)
    nccl_transpose_y_to_z!(storage, nccl_solver.nccl_yz)

    parent(solver.source_term) .= parent(storage.zfield)
    solve!(storage.zfield, solver.batched_tridiagonal_solver, solver.source_term)

    nccl_transpose_z_to_y!(storage, nccl_solver.nccl_yz)
    nccl_transpose_y_to_x!(storage, nccl_solver.nccl_xy)
    solver.plan.backward.x!(parent(storage.xfield), buffer.x)
    nccl_transpose_x_to_y!(storage, nccl_solver.nccl_xy)
    solver.plan.backward.y!(parent(storage.yfield), buffer.y)
    nccl_transpose_y_to_z!(storage, nccl_solver.nccl_yz)

    launch!(arch, solver.local_grid, :xyz,
            DC._copy_real_component!, x, parent(storage.zfield))

    return x
end

# ─── Y-stretched solve ───

function Oceananigans.Solvers.solve!(x, nccl_solver::NCCLYStretchedSolver)
    solver  = nccl_solver.solver
    arch    = DC.architecture(solver)
    storage = solver.storage
    buffer  = solver.buffer

    solver.plan.forward.z!(parent(storage.zfield), buffer.z)
    nccl_transpose_z_to_y!(storage, nccl_solver.nccl_yz)
    nccl_transpose_y_to_x!(storage, nccl_solver.nccl_xy)
    solver.plan.forward.x!(parent(storage.xfield), buffer.x)
    nccl_transpose_x_to_y!(storage, nccl_solver.nccl_xy)

    parent(solver.source_term) .= parent(storage.yfield)
    solve!(storage.yfield, solver.batched_tridiagonal_solver, solver.source_term)

    nccl_transpose_y_to_x!(storage, nccl_solver.nccl_xy)
    solver.plan.backward.x!(parent(storage.xfield), buffer.x)
    nccl_transpose_x_to_y!(storage, nccl_solver.nccl_xy)
    nccl_transpose_y_to_z!(storage, nccl_solver.nccl_yz)
    solver.plan.backward.z!(parent(storage.zfield), buffer.z)

    launch!(arch, solver.local_grid, :xyz,
            DC._copy_real_component!, x, parent(storage.zfield))

    return x
end

# ─── X-stretched solve ───

function Oceananigans.Solvers.solve!(x, nccl_solver::NCCLXStretchedSolver)
    solver  = nccl_solver.solver
    arch    = DC.architecture(solver)
    storage = solver.storage
    buffer  = solver.buffer

    solver.plan.forward.z!(parent(storage.zfield), buffer.z)
    nccl_transpose_z_to_y!(storage, nccl_solver.nccl_yz)
    solver.plan.forward.y!(parent(storage.yfield), buffer.y)
    nccl_transpose_y_to_x!(storage, nccl_solver.nccl_xy)

    parent(solver.source_term) .= parent(storage.xfield)
    solve!(storage.xfield, solver.batched_tridiagonal_solver, solver.source_term)

    nccl_transpose_x_to_y!(storage, nccl_solver.nccl_xy)
    solver.plan.backward.y!(parent(storage.yfield), buffer.y)
    nccl_transpose_y_to_z!(storage, nccl_solver.nccl_yz)
    solver.plan.backward.z!(parent(storage.zfield), buffer.z)

    launch!(arch, solver.local_grid, :xyz,
            DC._copy_real_component!, x, parent(storage.zfield))

    return x
end
