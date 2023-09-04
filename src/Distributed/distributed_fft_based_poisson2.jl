import FFTW 

import Oceananigans.Solvers: poisson_eigenvalues, solve!
import Oceananigans.Architectures: architecture
import Oceananigans.Fields: interior

struct DistributedFFTBasedPoissonSolver{P, F, L, λ, B, S}
    plan :: P
    global_grid :: F
    local_grid :: L
    eigenvalues :: λ
    buffer :: B
    storage :: S
end

architecture(solver::DistributedFFTBasedPoissonSolver) =
    architecture(solver.global_grid)

function DistributedFFTBasedPoissonSolver(global_grid, local_grid, planner_flag=FFTW.PATIENT)

    storage = ParallelFields(CenterField(local_grid), Complex{eltype(local_grid)})
    # We don't support distributing anything in z.
    architecture(local_grid).ranks[3] == 1 || throw(ArgumentError("Non-singleton ranks in the vertical are not supported by DistributedFFTBasedPoissonSolver."))

    arch = architecture(storage.xfield.grid)

    # Build _global_ eigenvalues
    topo = (TX, TY, TZ) = topology(global_grid)
    λx = poisson_eigenvalues(global_grid.Nx, global_grid.Lx, 1, TX())
    λy = partition_global_array(arch, poisson_eigenvalues(global_grid.Ny, global_grid.Ly, 2, TY()), (1, size(storage.xfield.grid, 2), 1))
    λz = partition_global_array(arch, poisson_eigenvalues(global_grid.Nz, global_grid.Lz, 3, TZ()), (1, 1, size(storage.xfield.grid, 3)))

    # Drop singleton dimensions for compatibility with PencilFFTs' localgrid
    λx = arch_array(arch, dropdims(λx, dims=(2, 3)))
    λy = arch_array(arch, dropdims(λy, dims=(1, 3)))
    λz = arch_array(arch, dropdims(λz, dims=(1, 2)))

    eigenvalues = (λx, λy, λz)

    plan = plan_distributed_transforms(global_grid, storage, planner_flag)

    child_arch = child_architecture(arch)

    buffer_x = child_arch isa GPU && TX == Bounded ? similar(storage.xfield) : nothing
    buffer_y = child_arch isa GPU && TX == Bounded ? similar(storage.yfield) : nothing
    buffer_z = child_arch isa GPU && TX == Bounded ? similar(storage.zfield) : nothing

    buffer = (x = buffer_x, y = buffer_y, z = buffer_z)

    return DistributedFFTBasedPoissonSolver(plan, global_grid, local_grid, eigenvalues, buffer, storage)
end

interior(::Nothing) = nothing

# solve! requires that `b` in `A x = b` (the right hand side)
# was computed and stored in first(solver.storage) prior to calling `solve!(x, solver)`.
# See: Models/NonhydrostaticModels/solve_for_pressure.jl
function solve!(x, solver::DistributedFFTBasedPoissonSolver)
    arch = architecture(solver.global_grid)
    multi_arch = architecture(solver.local_grid)

    # Apply forward transforms to b = first(solver.storage).
    solver.plan.forward[1](interior(solver.storage.zfield), interior(buffer.x))
    solver.plan.forward[2](solver.storage)
    solver.plan.forward[3](interior(solver.storage.yfield), interior(buffer.y))
    solver.plan.forward[4](solver.storage)
    solver.plan.forward[5](interior(solver.storage.xfield), interior(buffer.z))

    # Solve the discrete Poisson equation in wavenumber space
    # for x̂. We solve for x̂ in place, reusing b̂.
    λ = solver.eigenvalues
    x̂ = b̂ = solver.storage.xfield
    @. x̂ = - b̂ / (λ[1] + λ[2] + λ[3])

    # Set the zeroth wavenumber and volume mean, which are undetermined
    # in the Poisson equation, to zero.
    if MPI.Comm_rank(multi_arch.communicator) == 0
        # This is an assumption: we *hope* PencilArrays allocates in this way
        parent(x̂)[1, 1, 1] = 0
    end

    # Apply backward transforms to x̂ = last(solver.storage).
    solver.plan.backward[1](interior(solver.storage.xfield), interior(buffer.x))
    solver.plan.backward[2](solver.storage)
    solver.plan.backward[3](interior(solver.storage.yfield), interior(buffer.y))
    solver.plan.backward[4](solver.storage)
    solver.plan.backward[5](interior(solver.storage.zfield), interior(buffer.z))

    # Copy the real component of xc to x.
    launch!(arch, solver.local_grid, :xyz,
            _copy_real_component!, x, parent(solver.storage.zfield))

    return x
end

@kernel function _copy_real_component!(ϕ, ϕc)
    i, j, k = @index(Global, NTuple)
    @inbounds ϕ[i, j, k] = real(ϕc[i, j, k])
end

