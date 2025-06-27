import FFTW

using CUDA: @allowscalar
using Oceananigans.Grids: XYZRegularRG, XYRegularRG, XZRegularRG, YZRegularRG

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

In the following, `Nx`, `Ny`, and `Nz` are the number of grid points of the **global** grid,
in the `x`, `y`, and `z` directions, while `Rx`, `Ry`, and `Rz` are the number of ranks in the
`x`, `y`, and `z` directions, respectively. Furthermore, 'pencil' decomposition refers to a domain
decomposed in two different directions (i.e., with `Rx != 1` and `Ry != 1`), while 'slab' decomposition
refers to a domain decomposed only in one direction, (i.e., with either `Rx == 1` or `Ry == 1`).
Additionally, `storage` indicates the `TransposableField` used for storing intermediate results;
see [`TransposableField`](@ref).

1. Three dimensional grids with pencil decompositions in ``(x, y)`` such the:
the `z` direction is local, `Ny ≥ Rx` and `Ny % Rx = 0`, and `Nz ≥ Ry` and `Nz % Ry = 0`.

2. Two dimensional grids decomposed in ``x`` where `Ny ≥ Rx` and `Ny % Rx = 0`.

!!! warning "Unsupported decompositions"
    _Any_ configuration decomposed in ``z`` direction is _not_ supported.
    Furthermore, any ``(x, y)`` decompositions other than the configurations mentioned above are also _not_ supported.

Algorithm for pencil decompositions
===================================

For pencil decompositions (useful for three-dimensional problems), there are three forward transforms,
three backward transforms, and four transpositions that require MPI communication.
In the algorithm below, the first dimension is always the local dimension. In our implementation we require
`Nz ≥ Ry` and `Nx ≥ Ry` with the additional constraint that `Nz % Ry = 0` and `Ny % Rx = 0`.

1. `storage.zfield`, partitioned over ``(x, y)`` is initialized with the `rhs` that is ``b``.
2. Transform along ``z``.
3  Transpose `storage.zfield` + communicate to `storage.yfield` partitioned into `(Rx, Ry)` processes in ``(x, z)``.
4. Transform along ``y``.
5. Transpose `storage.yfield` + communicate to `storage.xfield` partitioned into `(Rx, Ry)` processes in ``(y, z)``.
6. Transform in ``x``.

At this point the three in-place forward transforms are complete, and we
solve the Poisson equation by updating `storage.xfield`.
Then the process is reversed to obtain `storage.zfield` in physical
space partitioned over ``(x, y)``.

Algorithm for stencil decompositions
====================================

The stecil decomposition algorithm works in the same manner as the pencil decompostion described above
while skipping the transposes that are not required. For example if the domain is decomposed in ``x``,
step 3 in the above algorithm is skipped (and the associated transposition step in the bakward transform)

Restrictions
============

1. Pencil decomopositions:
    - `Ny ≥ Rx` and `Ny % Rx = 0`
    - `Nz ≥ Ry` and `Nz % Ry = 0`
    - If the ``z`` direction is `Periodic`, also the ``y`` and the ``x`` directions must be `Periodic`
    - If the ``y`` direction is `Periodic`, also the ``x`` direction must be `Periodic`

2. Stencil decomposition:
    - same as for pencil decompositions with `Rx` (or `Ry`) equal to one
"""
function DistributedFFTBasedPoissonSolver(global_grid, local_grid, planner_flag=FFTW.PATIENT)

    validate_poisson_solver_distributed_grid(global_grid)
    validate_poisson_solver_configuration(global_grid, local_grid)

    FT = Complex{eltype(local_grid)}

    storage = TransposableField(CenterField(local_grid), FT)

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
validate_poisson_solver_distributed_grid(global_grid) =
        throw("Grids other than the RectilinearGrid are not supported in the Distributed NonhydrostaticModels")

function validate_poisson_solver_distributed_grid(global_grid::RectilinearGrid)
    TX, TY, TZ = topology(global_grid)

    if (TY == Bounded && TZ == Periodic) || (TX == Bounded && TY == Periodic) || (TX == Bounded && TZ == Periodic)
        throw("Distributed Poisson solvers do not support grids with topology ($TX, $TY, $TZ) at the moment.
               A Periodic z-direction requires also the y- and and x-directions to be Periodic, while a Periodic y-direction requires also
               the x-direction to be Periodic.")
    end

    if !(global_grid isa YZRegularRG) && !(global_grid isa XYRegularRG) && !(global_grid isa XZRegularRG)
        throw("The provided grid is stretched in directions $(stretched_dimensions(global_grid)).
               A distributed Poisson solver supports only RectilinearGrids that have variably-spaced cells in at most one direction.")
    end

    return nothing
end

function validate_poisson_solver_configuration(global_grid, local_grid)

    # We don't support distributing anything in z.
    Rx, Ry, Rz = architecture(local_grid).ranks
    Rz == 1 || throw("Non-singleton ranks in the vertical are not supported by distributed Poisson solvers.")

    # Limitation of the current implementation (see the docstring)
    if global_grid.Nz % Ry != 0
        throw("The number of ranks in the y-direction are $(Ry) with Nz = $(global_grid.Nz) cells in the z-direction.
               The distributed Poisson solver requires that the number of ranks in the y-direction divide Nz.")
    end

    if global_grid.Ny % Rx != 0
        throw("The number of ranks in the y-direction are $(Rx) with Ny = $(global_grid.Ny) cells in the y-direction.
               The distributed Poisson solver requires that the number of ranks in the x-direction divide Ny.")
    end

    return nothing
end
