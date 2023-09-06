import FFTW 

using CUDA: @allowscalar
using Oceananigans.Grids: YZRegRectilinearGrid

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

    validate_global_grid(global_grid)

    storage = ParallelFields(CenterField(local_grid), Complex{eltype(local_grid)})
    # We don't support distributing anything in z.
    architecture(local_grid).ranks[3] == 1 || throw(ArgumentError("Non-singleton ranks in the vertical are not supported by DistributedFFTBasedPoissonSolver."))

    arch = architecture(storage.xfield.grid)

    # Build _global_ eigenvalues
    topo = (TX, TY, TZ) = topology(global_grid)
    λx = dropdims(poisson_eigenvalues(global_grid.Nx, global_grid.Lx, 1, TX()), dims=(2, 3))
    λy = dropdims(poisson_eigenvalues(global_grid.Ny, global_grid.Ly, 2, TY()), dims=(1, 3))
    λz = dropdims(poisson_eigenvalues(global_grid.Nz, global_grid.Lz, 3, TZ()), dims=(1, 2))
        
    λx = partition(λx, size(storage.xfield.grid, 1), arch, 1)
    λy = partition(λy, size(storage.xfield.grid, 2), arch, 2)
    λz = partition(λz, size(storage.xfield.grid, 3), arch, 3)

    λx = arch_array(arch, λx)
    λy = arch_array(arch, λy)
    λz = arch_array(arch, λz)

    eigenvalues = (λx, λy, λz)

    plan = plan_distributed_transforms(global_grid, storage, planner_flag)
    buffer = parent(similar(storage.yfield)) # We cannot really batch anything, so always reshape

    return DistributedFFTBasedPoissonSolver(plan, global_grid, local_grid, eigenvalues, buffer, storage)
end

# solve! requires that `b` in `A x = b` (the right hand side)
# was computed and stored in first(solver.storage) prior to calling `solve!(x, solver)`.
# See: Models/NonhydrostaticModels/solve_for_pressure.jl
function solve!(x, solver::DistributedFFTBasedPoissonSolver)
    storage = solver.storage
    buffer  = solver.buffer

    arch    = architecture(storage.xfield.grid)

    # Apply forward transforms to b = first(solver.storage).
    solver.plan.forward.z!(parent(storage.zfield), nothing)
    transpose_z_to_y!(storage)
    solver.plan.forward.y!(parent(storage.yfield), nothing) 
    transpose_y_to_x!(storage)
    solver.plan.forward.x!(parent(storage.xfield), nothing)
    
    # Solve the discrete Poisson equation in wavenumber space
    # for x̂. We solve for x̂ in place, reusing b̂.
    λ = solver.eigenvalues
    x̂ = b̂ = parent(storage.xfield)

    launch!(arch, storage.xfield.grid, :xyz, _solve_poisson!, x̂, b̂, λ[1], λ[2], λ[3])

    # Set the zeroth wavenumber and volume mean, which are undetermined
    # in the Poisson equation, to zero.
    if arch.local_rank == 0
        @allowscalar x̂[1, 1, 1] = 0
    end

    # Apply backward transforms to x̂ = last(solver.storage).
    solver.plan.backward.x!(parent(storage.xfield), nothing)
    transpose_x_to_y!(storage)
    solver.plan.backward.y!(parent(storage.yfield), nothing)
    transpose_y_to_z!(storage)
    solver.plan.backward.z!(parent(storage.zfield), nothing)

    # Copy the real component of xc to x.
    launch!(arch, solver.local_grid, :xyz,
            _copy_real_component!, x, parent(storage.zfield))

    return x
end

@kernel function _solve_poisson!(x̂, b̂, λx, λy, λz)
    i, j, k = @index(Global, NTuple)
    @inbounds x̂[i, j, k] = - b̂[i, j, k] / (λx[i] + λy[j] + λz[k])
end

@kernel function _copy_real_component!(ϕ, ϕc)
    i, j, k = @index(Global, NTuple)
    @inbounds ϕ[i, j, k] = real(ϕc[i, j, k])
end

validate_global_grid(global_grid) = 
        throw(ArgumentError("Grids other than the RectilinearGrid are not supported in the Distributed NonhydrostaticModels"))

function validate_global_grid(global_grid::RectilinearGrid) 
    TX, TY, TZ = topology(global_grid)

    if (TY == Bounded && TZ == Periodic) || (TX == Bounded && TY == Periodic) || (TX == Bounded && TZ == Periodic)
        throw(ArgumentError("NonhydrostaticModels on Distributed grids do not support topology ($TX, $TY, $TZ).
                             For performance reasons, TZ Periodic requires also TY and TX to be Periodic,
                             while TY Periodic requires also TX to be Periodic. 
                             Please rotate the domain to obtain the required topology"))
    end
    
    if !(global_grid isa YZRegRectilinearGrid) 
        throw(ArgumentError("For performance reasons only stretching on the X direction is allowed with 
                             distributed grids. Please rotate the domain to have the stretching in X"))
    end

    return nothing
end