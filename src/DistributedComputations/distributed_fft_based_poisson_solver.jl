import FFTW 

using CUDA: @allowscalar
using Oceananigans.Grids: YZRegularRG

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

"""
    DistributedFFTBasedPoissonSolver(global_grid, local_grid)

Return an FFT-based solver for the Poisson equation,

```math
∇²φ = b
```

for `Distributed` architectures.

Supported configurations
========================

1. Three dimensional configurations with two-dimensional decompositions in ``(x, y)`` 
where `Ny ≥ Rx` and `Ny % Rx = 0`, and `Nz ≥ Ry` and `Nz % Ry = 0`.

2. Two dimensional configurations decomposed in ``x`` where `Ny ≥ Rx` and `Ny % Rx = 0`
    
Other configurations that are decomposed in ``(x, y)``,
or any configuration decomposed in ``z``, are _not_ supported.

Algorithm for two-dimensional decompositions
============================================

For two-dimensional decompositions (useful for three-dimensional problems),
there are three forward transforms, three backward transforms,
and four transpositions requiring MPI communication. In the schematic below, the first
dimension is always the local dimension. In our implementation we require
`Nz ≥ Ry` and `Nx ≥ Ry` with the additional constraint that `Nz % Ry = 0` and `Ny % Rx = 0`.
`Rx` is the number of ranks in ``x``, and `Ry` is the number of ranks in ``y``.

1. `storage.zfield`, partitioned over ``(x, y)`` is initialized with the `rhs`.
2. Transform along ``z``.
3  Transpose + communicate to `storage.yfield` partitioned into `(Rx, Ry)` processes in ``(x, z)``.
4. Transform along ``y``.
5. Transpose + communicate to `storage.xfield` partitioned into `(Rx, Ry)` processes in ``(y, z)``.
6. Transform in ``x``.

At this point the three in-place forward transforms are complete, and we
solve the Poisson equation by updating `storage.xfield`.
Then the process is reversed to obtain `storage.zfield` in physical
space partitioned over ``(x, y)``.

Algorithm for one-dimensional decompositions
============================================

The one-dimensional decomposition works in the same manner while skipping the transposes that
are not required. For example if the domain is decomposed in ``x``, step 3 in the above algorithm
is skipped (and the associated transposition step in the bakward transform)

Restrictions
============

1. Two-dimensional decomopositions:
    - `Ny ≥ Rx` and `Ny % Rx = 0`
    - `Nz ≥ Ry` and `Nz % Ry = 0`
    - If the ``z`` direction is `Periodic`, also the ``y`` and the ``x`` directions must be `Periodic`
    - If the ``y`` direction is `Periodic`, also the ``x`` direction must be `Periodic`

2. One-dimensional decomposition:
    - same as for two-dimensional decompositions with `Rx` (or `Ry`) equal to one

"""
function DistributedFFTBasedPoissonSolver(global_grid, local_grid, planner_flag=FFTW.PATIENT)

    validate_global_grid(global_grid)
    FT = Complex{eltype(local_grid)}

    storage = TransposableField(CenterField(local_grid), FT)
    # We don't support distributing anything in z.
    Rz = architecture(local_grid).ranks[3]
    Rz == 1 || throw(ArgumentError("Non-singleton ranks in the vertical are not supported by DistributedFFTBasedPoissonSolver."))

    arch = architecture(storage.xfield.grid)
    child_arch = child_architecture(arch)

    # Build _global_ eigenvalues
    topo = (TX, TY, TZ) = topology(global_grid)
    λx = dropdims(poisson_eigenvalues(global_grid.Nx, global_grid.Lx, 1, TX()), dims=(2, 3))
    λy = dropdims(poisson_eigenvalues(global_grid.Ny, global_grid.Ly, 2, TY()), dims=(1, 3))
    λz = dropdims(poisson_eigenvalues(global_grid.Nz, global_grid.Lz, 3, TZ()), dims=(1, 2))
        
    λx = partition_coordinate(λx, size(storage.xfield.grid, 1), arch, 1)
    λy = partition_coordinate(λy, size(storage.xfield.grid, 2), arch, 2)
    λz = partition_coordinate(λz, size(storage.xfield.grid, 3), arch, 3)

    λx = on_architecture(child_arch, λx)
    λy = on_architecture(child_arch, λy)
    λz = on_architecture(child_arch, λz)

    eigenvalues = (λx, λy, λz)

    plan = plan_distributed_transforms(global_grid, storage, planner_flag)
    
    # We need to permute indices to apply bounded transforms on the GPU (r2r of r2c with twiddling)
    x_buffer_needed = child_arch isa GPU && TX == Bounded
    z_buffer_needed = child_arch isa GPU && TZ == Bounded 
    
    # We cannot really batch anything, so on GPUs we always have to permute indices in the y direction
    y_buffer_needed = child_arch isa GPU

    buffer_x = x_buffer_needed ? on_architecture(child_arch, zeros(FT, size(storage.xfield)...)) : nothing
    buffer_y = y_buffer_needed ? on_architecture(child_arch, zeros(FT, size(storage.yfield)...)) : nothing 
    buffer_z = z_buffer_needed ? on_architecture(child_arch, zeros(FT, size(storage.zfield)...)) : nothing

    buffer = (; x = buffer_x, y = buffer_y, z = buffer_z)

    return DistributedFFTBasedPoissonSolver(plan, global_grid, local_grid, eigenvalues, buffer, storage)
end

# solve! requires that `b` in `A x = b` (the right hand side) 
# is copied in the solver storage
# See: Models/NonhydrostaticModels/solve_for_pressure.jl
function solve!(x, solver::DistributedFFTBasedPoissonSolver)
    storage = solver.storage
    buffer  = solver.buffer
    arch    = architecture(storage.xfield.grid)

    # Apply forward transforms to b = first(solver.storage).
    solver.plan.forward.z!(parent(storage.zfield), buffer.z)
    transpose_z_to_y!(storage) # copy data from storage.zfield to storage.yfield
    solver.plan.forward.y!(parent(storage.yfield), buffer.y) 
    transpose_y_to_x!(storage) # copy data from storage.yfield to storage.xfield
    solver.plan.forward.x!(parent(storage.xfield), buffer.x)
    
    # Solve the discrete Poisson equation in wavenumber space
    # for x̂. We solve for x̂ in place, reusing b̂.
    λ = solver.eigenvalues
    x̂ = b̂ = parent(storage.xfield)
    
    launch!(arch, storage.xfield.grid, :xyz, _solve_poisson_in_spectral_space!, x̂, b̂, λ[1], λ[2], λ[3])

    # Set the zeroth wavenumber and volume mean, which are undetermined
    # in the Poisson equation, to zero.
    if arch.local_rank == 0
        @allowscalar x̂[1, 1, 1] = 0
    end

    # Apply backward transforms to x̂ = parent(storage.xfield).
    solver.plan.backward.x!(parent(storage.xfield), buffer.x)
    transpose_x_to_y!(storage) # copy data from storage.xfield to storage.yfield
    solver.plan.backward.y!(parent(storage.yfield), buffer.y)
    transpose_y_to_z!(storage) # copy data from storage.yfield to storage.zfield
    solver.plan.backward.z!(parent(storage.zfield), buffer.z) # last backwards transform is in z

    # Copy the real component of xc to x.
    launch!(arch, solver.local_grid, :xyz,
            _copy_real_component!, x, parent(storage.zfield))

    return x
end

@kernel function _solve_poisson_in_spectral_space!(x̂, b̂, λx, λy, λz)
    i, j, k = @index(Global, NTuple)
    @inbounds x̂[i, j, k] = - b̂[i, j, k] / (λx[i] + λy[j] + λz[k])
end

@kernel function _copy_real_component!(ϕ, ϕc)
    i, j, k = @index(Global, NTuple)
    @inbounds ϕ[i, j, k] = real(ϕc[i, j, k])
end

# TODO: bring up to speed the PCG to remove this error
validate_global_grid(global_grid) = 
        throw(ArgumentError("Grids other than the RectilinearGrid are not supported in the Distributed NonhydrostaticModels"))

function validate_global_grid(global_grid::RectilinearGrid) 
    TX, TY, TZ = topology(global_grid)

    if (TY == Bounded && TZ == Periodic) || (TX == Bounded && TY == Periodic) || (TX == Bounded && TZ == Periodic)
        throw(ArgumentError("NonhydrostaticModels on Distributed grids do not support topology ($TX, $TY, $TZ) at the moment.
                             TZ Periodic requires also TY and TX to be Periodic,
                             while TY Periodic requires also TX to be Periodic. 
                             Please rotate the domain to obtain the required topology"))
    end
    
    # TODO: Allow stretching in z by rotating the underlying data in order to 
    # have just 4 transposes as opposed to 8    
    if !(global_grid isa YZRegularRG) 
        throw(ArgumentError("Only stretching on the X direction is supported with distributed grids at the moment. 
                             Please rotate the domain to have the stretching in X"))
    end

    return nothing
end

