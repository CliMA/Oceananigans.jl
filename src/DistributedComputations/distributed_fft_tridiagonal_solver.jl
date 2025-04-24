using CUDA: @allowscalar
using Oceananigans.Grids: stretched_dimensions
using Oceananigans.Grids: XDirection, YDirection
using Oceananigans.Operators: Δxᶠᵃᵃ, Δyᵃᶠᵃ, Δzᵃᵃᶠ

using Oceananigans.Solvers: BatchedTridiagonalSolver,
                            stretched_direction,
                            ZTridiagonalSolver,
                            YTridiagonalSolver,
                            XTridiagonalSolver,
                            compute_main_diagonal!

struct DistributedFourierTridiagonalPoissonSolver{G, L, B, P, R, S, β}
    plan :: P
    global_grid :: G
    local_grid :: L
    batched_tridiagonal_solver :: B
    source_term :: R
    storage :: S
    buffer :: β
end

# Usefull aliases for dispatch...
const XStretchedDistributedSolver = DistributedFourierTridiagonalPoissonSolver{<:Any, <:Any, <:XTridiagonalSolver}
const YStretchedDistributedSolver = DistributedFourierTridiagonalPoissonSolver{<:Any, <:Any, <:YTridiagonalSolver}
const ZStretchedDistributedSolver = DistributedFourierTridiagonalPoissonSolver{<:Any, <:Any, <:ZTridiagonalSolver}

architecture(solver::DistributedFourierTridiagonalPoissonSolver) =
    architecture(solver.global_grid)

@inline Δξᶠ(i, grid, ::Val{1}) = Δxᶠᵃᵃ(i, 1, 1, grid)
@inline Δξᶠ(j, grid, ::Val{2}) = Δyᵃᶠᵃ(1, j, 1, grid)
@inline Δξᶠ(k, grid, ::Val{3}) = Δzᵃᵃᶠ(1, 1, k, grid)

"""
    DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid)

Return an FFT-based solver for the Poisson equation evaluated on a `local_grid` that has a non-uniform
spacing in exactly one direction (i.e. either in x, y or z)

```math
∇²φ = b
```

for `Distributed` architectures.

Supported configurations
========================

In the following, `Nx`, `Ny`, and `Nz` are the number of grid points of the **global** grid
in the `x`, `y`, and `z` directions, while `Rx`, `Ry`, and `Rz` are the number of ranks in the
`x`, `y`, and `z` directions, respectively. Furthermore, 'pencil' decomposition refers to a domain
decomposed in two different directions (i.e., with `Rx != 1` and `Ry != 1`), while 'slab' decomposition
refers to a domain decomposed only in one direction, (i.e., with either `Rx == 1` or `Ry == 1`).
Additionally, `storage` indicates the `TransposableField` used for storing intermediate results;
see [`TransposableField`](@ref).

1. Three dimensional configurations with 'pencil' decompositions in ``(x, y)``
where `Ny ≥ Rx` and `Ny % Rx = 0`, and `Nz ≥ Ry` and `Nz % Ry = 0`.

2. Two dimensional configurations decomposed in ``x`` where `Ny ≥ Rx` and `Ny % Rx = 0`

!!! warning "Unsupported decompositions"
    _Any_ configuration decomposed in ``z`` direction is _not_ supported.
    Furthermore, any ``(x, y)`` decompositions other than the configurations mentioned above are also _not_ supported.

Algorithm for pencil decompositions
============================================

For pencil decompositions (useful for three-dimensional problems),
there are two forward transforms, two backward transforms, one tri-diagonal matrix inversion
and a variable number of transpositions that require MPI communication, dependent on the
stretched direction:

- a stretching in the x-direction requires four transpositions
- a stretching in the y-direction requires six transpositions
- a stretching in the z-direction requires eight transpositions

!!! note "Computational cost"
    Because of the additional transpositions, a stretching in the x-direction
    is computationally cheaper than a stretching in the y-direction, and the latter
    is cheaper than a stretching in the z-direction

In our implementation we require `Nz ≥ Ry` and `Nx ≥ Ry` with the additional constraint
that `Nz % Ry = 0` and `Ny % Rx = 0`.

x - stretched algorithm
========================

1. `storage.zfield`, partitioned over ``(x, y)`` is initialized with the `rhs`.
2. Transform along ``z``.
3. Transpose `storage.zfield` + communicate to `storage.yfield` partitioned into `(Rx, Ry)` processes in ``(x, z)``.
4. Transform along ``y``.
5. Transpose `storage.yfield` + communicate to `storage.xfield` partitioned into `(Rx, Ry)` processes in ``(y, z)``.
6. Solve the tri-diagonal linear system in the ``x`` direction.

Steps 5 -> 1 are reversed to obtain the result in physical space stored in `storage.zfield`
partitioned over ``(x, y)``.

y - stretched algorithm
========================

1. `storage.zfield`, partitioned over ``(x, y)`` is initialized with the `rhs`.
2. Transform along ``z``.
3. Transpose `storage.zfield` + communicate to `storage.yfield` partitioned into `(Rx, Ry)` processes in ``(x, z)``.
4. Transpose `storage.yfield` + communicate to `storage.xfield` partitioned into `(Rx, Ry)` processes in ``(y, z)``.
5. Transform along ``x``.
6. Transpose `storage.xfield` + communicate to `storage.yfield` partitioned into `(Rx, Ry)` processes in ``(x, z)``.
7. Solve the tri-diagonal linear system in the ``y`` direction.

Steps 6 -> 1 are reversed to obtain the result in physical space stored in `storage.zfield`
partitioned over ``(x, y)``.

z - stretched algorithm
========================

1. `storage.zfield`, partitioned over ``(x, y)`` is initialized with the `rhs`.
2. Transpose `storage.zfield` + communicate to `storage.yfield` partitioned into `(Rx, Ry)` processes in ``(x, z)``.
3. Transform along ``y``.
4. Transpose `storage.yfield` + communicate to `storage.xfield` partitioned into `(Rx, Ry)` processes in ``(y, z)``.
5. Transform along ``x``.
6. Transpose `storage.xfield` + communicate to `storage.yfield` partitioned into `(Rx, Ry)` processes in ``(x, z)``.
7. Transpose `storage.yfield` + communicate to `storage.zfield` partitioned into `(Rx, Ry)` processes in ``(x, y)``.
8. Solve the tri-diagonal linear system in the ``z`` direction.

Steps 7 -> 1 are reversed to obtain the result in physical space stored in `storage.zfield`
partitioned over ``(x, y)``.

Algorithm for slab decompositions
=============================

The 'slab' decomposition works in the same manner while skipping the transposes that
are not required. For example if the domain is decomposed in ``x``, step 4. and 6. in the above algorithm
are skipped (and the associated reversed step in the backward transform)

Restrictions
============

1. Pencil decompositions:
    - `Ny ≥ Rx` and `Ny % Rx = 0`
    - `Nz ≥ Ry` and `Nz % Ry = 0`
    - If the ``z`` direction is `Periodic`, also the ``y`` and the ``x`` directions must be `Periodic`
    - If the ``y`` direction is `Periodic`, also the ``x`` direction must be `Periodic`

2. Slab decomposition:
    - Same as for two-dimensional decompositions with `Rx` (or `Ry`) equal to one

"""
function DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid, planner_flag=FFTW.PATIENT; tridiagonal_direction = nothing)

    validate_poisson_solver_distributed_grid(global_grid)
    validate_poisson_solver_configuration(global_grid, local_grid)

    if isnothing(tridiagonal_direction)
        tridiagonal_dim = stretched_dimensions(local_grid)[1]
        tridiagonal_direction = stretched_direction(local_grid)
    else
        tridiagonal_dim = tridiagonal_direction == XDirection() ? 1 :
                          tridiagonal_direction == YDirection() ? 2 : 3
    end

    topology(global_grid, tridiagonal_dim) != Bounded &&
        error("`DistributedFourierTridiagonalPoissonSolver` requires that the stretched direction (dimension $tridiagonal_dim) is `Bounded`.")

    FT         = Complex{eltype(local_grid)}
    child_arch = child_architecture(local_grid)
    storage    = TransposableField(CenterField(local_grid), FT)

    topo = (TX, TY, TZ) = topology(global_grid)
    λx = dropdims(poisson_eigenvalues(global_grid.Nx, global_grid.Lx, 1, TX()), dims=(2, 3))
    λy = dropdims(poisson_eigenvalues(global_grid.Ny, global_grid.Ly, 2, TY()), dims=(1, 3))
    λz = dropdims(poisson_eigenvalues(global_grid.Nz, global_grid.Lz, 3, TZ()), dims=(1, 2))

    if tridiagonal_dim == 1
        arch = architecture(storage.xfield.grid)
        grid = storage.xfield.grid
        λ1 = partition_coordinate(λy, size(grid, 2), arch, 2)
        λ2 = partition_coordinate(λz, size(grid, 3), arch, 3)
    elseif tridiagonal_dim == 2
        arch = architecture(storage.yfield.grid)
        grid = storage.yfield.grid
        λ1 = partition_coordinate(λx, size(grid, 1), arch, 1)
        λ2 = partition_coordinate(λz, size(grid, 3), arch, 3)
    elseif tridiagonal_dim == 3
        arch = architecture(storage.zfield.grid)
        grid = storage.zfield.grid
        λ1 = partition_coordinate(λx, size(grid, 1), arch, 1)
        λ2 = partition_coordinate(λy, size(grid, 2), arch, 2)
    end

    λ1 = on_architecture(child_arch, λ1)
    λ2 = on_architecture(child_arch, λ2)

    plan = plan_distributed_transforms(global_grid, storage, planner_flag)

    # Lower and upper diagonals are the same
    lower_diagonal = @allowscalar [ 1 / Δξᶠ(q, grid, Val(tridiagonal_dim)) for q in 2:size(grid, tridiagonal_dim) ]
    lower_diagonal = on_architecture(child_arch, lower_diagonal)
    upper_diagonal = lower_diagonal

    # Compute diagonal coefficients for each grid point
    diagonal = zeros(eltype(grid), size(grid)...)
    diagonal = on_architecture(arch, diagonal)
    launch_config = if tridiagonal_dim == 1
                        :yz
                    elseif tridiagonal_dim == 2
                        :xz
                    elseif tridiagonal_dim == 3
                        :xy
                    end

    launch!(arch, grid, launch_config, compute_main_diagonal!, diagonal, grid, λ1, λ2, tridiagonal_direction)

    # Set up batched tridiagonal solver
    btsolver = BatchedTridiagonalSolver(grid; lower_diagonal, diagonal, upper_diagonal, tridiagonal_direction)

    # We need to permute indices to apply bounded transforms on the GPU (r2r of r2c with twiddling)
    x_buffer_needed = child_arch isa GPU && TX == Bounded
    z_buffer_needed = child_arch isa GPU && TZ == Bounded

    # We cannot really batch anything, so on GPUs we always have to permute indices in the y direction
    y_buffer_needed = child_arch isa GPU

    buffer_x = x_buffer_needed ? on_architecture(child_arch, zeros(FT, size(storage.xfield)...)) : nothing
    buffer_y = y_buffer_needed ? on_architecture(child_arch, zeros(FT, size(storage.yfield)...)) : nothing
    buffer_z = z_buffer_needed ? on_architecture(child_arch, zeros(FT, size(storage.zfield)...)) : nothing

    buffer = if tridiagonal_dim == 1
        (; y = buffer_y, z = buffer_z)
    elseif tridiagonal_dim == 2
        (; x = buffer_x, z = buffer_z)
    elseif tridiagonal_dim == 3
        (; x = buffer_x, y = buffer_y)
    end

    if tridiagonal_dim == 1
        forward  = (y! = plan.forward.y!,  z! = plan.forward.z!)
        backward = (y! = plan.backward.y!, z! = plan.backward.z!)
    elseif tridiagonal_dim == 2
        forward  = (x! = plan.forward.x!,  z! = plan.forward.z!)
        backward = (x! = plan.backward.x!, z! = plan.backward.z!)
    elseif tridiagonal_dim == 3
        forward  = (x! = plan.forward.x!,  y! = plan.forward.y!)
        backward = (x! = plan.backward.x!, y! = plan.backward.y!)
    end

    plan = (; forward, backward)

    # Storage space for right hand side of Poisson equation
    T = complex(eltype(grid))
    source_term = zeros(T, size(grid)...)
    source_term = on_architecture(arch, source_term)

    return DistributedFourierTridiagonalPoissonSolver(plan, global_grid, local_grid, btsolver, source_term, storage, buffer)
end

# solve! requires that `b` in `A x = b` (the right hand side)
# is copied in the solver storage
# See: Models/NonhydrostaticModels/solve_for_pressure.jl
function solve!(x, solver::ZStretchedDistributedSolver)
    arch    = architecture(solver)
    storage = solver.storage
    buffer  = solver.buffer

    transpose_z_to_y!(storage) # copy data from storage.zfield to storage.yfield
    solver.plan.forward.y!(parent(storage.yfield), buffer.y)
    transpose_y_to_x!(storage) # copy data from storage.yfield to storage.xfield
    solver.plan.forward.x!(parent(storage.xfield), buffer.x)
    transpose_x_to_y!(storage) # copy data from storage.xfield to storage.yfield
    transpose_y_to_z!(storage) # copy data from storage.yfield to storage.zfield

    # copy results in the source term
    parent(solver.source_term) .= parent(storage.zfield)

    # Perform the implicit vertical solve here on storage.zfield...
    # Solve tridiagonal system of linear equations at every z-column.
    solve!(storage.zfield, solver.batched_tridiagonal_solver, solver.source_term)

    transpose_z_to_y!(storage)
    transpose_y_to_x!(storage) # copy data from storage.yfield to storage.xfield
    solver.plan.backward.x!(parent(storage.xfield), buffer.x)
    transpose_x_to_y!(storage) # copy data from storage.xfield to storage.yfield
    solver.plan.backward.y!(parent(storage.yfield), buffer.y)
    transpose_y_to_z!(storage) # copy data from storage.yfield to storage.zfield

    # Copy the real component of xc to x.
    launch!(arch, solver.local_grid, :xyz,
            _copy_real_component!, x, parent(storage.zfield))

    return x
end

function solve!(x, solver::YStretchedDistributedSolver)
    arch    = architecture(solver)
    storage = solver.storage
    buffer  = solver.buffer

    solver.plan.forward.z!(parent(storage.zfield), buffer.z)
    transpose_z_to_y!(storage) # copy data from storage.zfield to storage.yfield
    transpose_y_to_x!(storage) # copy data from storage.yfield to storage.xfield
    solver.plan.forward.x!(parent(storage.xfield), buffer.x)
    transpose_x_to_y!(storage) # copy data from storage.xfield to storage.yfield

    # copy results in the source term
    parent(solver.source_term) .= parent(storage.yfield)

    # Perform the implicit vertical solve here on storage.yfield...
    # Solve tridiagonal system of linear equations at every y-column.
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

function solve!(x, solver::XStretchedDistributedSolver)
    arch    = architecture(solver)
    storage = solver.storage
    buffer  = solver.buffer

    # Apply forward transforms to b = first(solver.storage).
    solver.plan.forward.z!(parent(storage.zfield), buffer.z)
    transpose_z_to_y!(storage) # copy data from storage.zfield to storage.yfield
    solver.plan.forward.y!(parent(storage.yfield), buffer.y)
    transpose_y_to_x!(storage) # copy data from storage.yfield to storage.xfield

    # copy results in the source term
    parent(solver.source_term) .= parent(storage.xfield)

    # Perform the implicit vertical solve here on storage.xfield...
    # Solve tridiagonal system of linear equations at every x-column.
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