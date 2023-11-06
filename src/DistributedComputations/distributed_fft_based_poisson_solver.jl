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

Return a FFT-based solver for the Poisson equation,

```math
∇²φ = b
```

for `Distributed` architectures.

Supported configurations
========================

We support two "modes":

  1. Vertical pencil decompositions: two-dimensional decompositions in ``(x, y)``
     for three dimensional problems that satisfy either `Nz > Rx` or `Nz > Ry`.

  2. One-dimensional decompositions in either ``x`` or ``y``.

Above, `Nz = size(global_grid, 3)` and `Rx, Ry, Rz = architecture(local_grid).ranks`.

Other configurations that are decomposed in ``(x, y)`` but have too few `Nz`,
or any configuration decomposed in ``z``, are _not_ supported.

Algorithm for two-dimensional decompositions
============================================

For two-dimensional decompositions (useful for three-dimensional problems),
there are three forward transforms, three backward transforms,
and four transpositions requiring MPI communication. In the schematic below, the first
dimension is always the local dimension. In our implementation of the PencilFFTs algorithm,
we require _either_ `Nz >= Rx`, _or_ `Nz >= Ry`, where `Nz` is the number of vertical cells,
`Rx` is the number of ranks in ``x``, and `Ry` is the number of ranks in ``y``.
Below, we outline the algorithm for the case `Nz >= Rx`.
If `Nz < Rx`, but `Nz > Ry`, a similar algorithm applies with ``x`` and ``y`` swapped:

1. `first(storage)` is initialized with layout ``(z, x, y)``.
2. Transform along ``z``.
3  Transpose + communicate to `storage[2]` in layout ``(x, z, y)``,
   which is distributed into `(Rx, Ry)` processes in ``(z, y)``.
4. Transform along ``x``.
5. Transpose + communicate to `last(storage)` in layout ``(y, x, z)``,
   which is distributed into `(Rx, Ry)` processes in ``(x, z)``.
6. Transform in ``y``.

At this point the three in-place forward transforms are complete, and we
solve the Poisson equation by updating `last(storage)`.
Then the process is reversed to obtain `first(storage)` in physical
space and with the layout ``(z, x, y)``.

Restrictions
============

The algorithm for two-dimensional decompositions requires that `Nz = size(global_grid, 3)` is larger
than either `Rx = ranks[1]` or `Ry = ranks[2]`, where `ranks` are configured when building `Distributed`.
If `Nz` does not satisfy this condition, we can only support a one-dimensional decomposition.

Algorithm for one-dimensional decompositions
============================================

This algorithm requires a one-dimensional decomposition with _either_ `Rx = 1`
_or_ `Ry = 1`, and is important to support two-dimensional transforms.

For one-dimensional decompositions, we place the decomposed direction _last_.
If the number of ranks is `Rh = max(Rx, Ry)`, this algorithm requires that 
_both_ `Nx > Rh` _and_ `Ny > Rh`. The resulting flow of transposes and transforms
is similar to the two-dimensional case. It remains somewhat of a mystery why this
succeeds (i.e., why the last transform is correctly decomposed).
"""
function DistributedFFTBasedPoissonSolver(global_grid, local_grid, planner_flag=FFTW.PATIENT)

    validate_global_grid(global_grid)
    FT = Complex{eltype(local_grid)}

    storage = ParallelFields(CenterField(local_grid), FT)
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

    plan   = plan_distributed_transforms(global_grid, storage, planner_flag)
    
    # We need to permute indices to apply bounded transforms on the GPU (r2r of r2c with twiddling)
    buffer_x = child_architecture(arch) isa GPU && TX == Bounded ? arch_array(arch, zeros(FT, size(storage.xfield)...)) : nothing
    buffer_z = child_architecture(arch) isa GPU && TZ == Bounded ? arch_array(arch, zeros(FT, size(storage.zfield)...)) : nothing
    # We cannot really batch anything, so on GPUs we always have to permute indices in the y direction
    buffer_y = child_architecture(arch) isa GPU ? arch_array(arch, zeros(FT, size(storage.yfield)...)) : nothing 

    buffer = (; x = buffer_x, y = buffer_y, z = buffer_z)

    return DistributedFFTBasedPoissonSolver(plan, global_grid, local_grid, eigenvalues, buffer, storage)
end

# solve! requires that `b` in `A x = b` (the right hand side)
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

    launch!(arch, storage.xfield.grid, :xyz, _solve_poisson!, x̂, b̂, λ[1], λ[2], λ[3])

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

@kernel function _solve_poisson!(x̂, b̂, λx, λy, λz)
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

