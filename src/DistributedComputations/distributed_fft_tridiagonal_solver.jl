using CUDA: @allowscalar
using Oceananigans.Solvers: BatchedTridiagonalSolver
using Oceananigans.Solvers: stretched_dimensions, stretched_direction
using Oceananigans.Solvers: Δξᶠ, compute_main_diagonal!

using Oceananigans.Grids: XZRegularRG, 
                          XYRegularRG,
                          YZRegularRG

struct DistributedFourierTridiagonalPoissonSolver{G, L, P, B, R, S, β} 
                          plan :: P              
                   global_grid :: G
                    local_grid :: L
    batched_tridiagonal_solver :: B
                   source_term :: R
                       storage :: S
                        buffer :: β
end

architecture(solver::DistributedFourierTridiagonalPoissonSolver) =
    architecture(solver.global_grid)

"""
    DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid)

Return an FFT-based solver for the Poisson equation evaluated on a `local_grid`
with one stretched direction

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
strethed direction:

- a stretching in the x-direction requires four transpositions
- a stretching in the y-direction requires six transpositions
- a stretching in the z-direction requires eight transpositions

!!! warning "Computational cost"
    Because of the additional transpositions, a stretching in the x-direction
    is computationally cheaper than a stretching in the y-direction, and the latter
    is cheaper than a stretching in the z-direction

In our implementation we require `Nz ≥ Ry` and `Nx ≥ Ry` with the additional constraint 
that `Nz % Ry = 0` and `Ny % Rx = 0`.

X - stretched algorithm
========================

1. `storage.zfield`, partitioned over ``(x, y)`` is initialized with the `rhs`.
2. Transform along ``z``.
3. Transpose + communicate to `storage.yfield` partitioned into `(Rx, Ry)` processes in ``(x, z)``.
4. Transform along ``y``.
5. Transpose + communicate to `storage.xfield` partitioned into `(Rx, Ry)` processes in ``(y, z)``.
6. Solve the tri-diagonal linear system in the ``x`` direction.

Steps 5 -> 1 are reversed to obtain the result in physical space stored in `storage.zfield` 
partitioned over ``(x, y)``.

Y - stretched algorithm
========================

1. `storage.zfield`, partitioned over ``(x, y)`` is initialized with the `rhs`.
2. Transform along ``z``.
3. Transpose + communicate to `storage.yfield` partitioned into `(Rx, Ry)` processes in ``(x, z)``.
4. Transpose + communicate to `storage.xfield` partitioned into `(Rx, Ry)` processes in ``(y, z)``.
5. Transform along ``x``.
6. Transpose + communicate to `storage.yfield` partitioned into `(Rx, Ry)` processes in ``(x, z)``.
7. Solve the tri-diagonal linear system in the ``y`` direction.

Steps 6 -> 1 are reversed to obtain the result in physical space stored in `storage.zfield` 
partitioned over ``(x, y)``.

Z - stretched algorithm
========================

1. `storage.zfield`, partitioned over ``(x, y)`` is initialized with the `rhs`.
2. Transpose + communicate to `storage.yfield` partitioned into `(Rx, Ry)` processes in ``(x, z)``.
3. Transform along ``y``.
4. Transpose + communicate to `storage.xfield` partitioned into `(Rx, Ry)` processes in ``(y, z)``.
5. Transform along ``x``.
6. Transpose + communicate to `storage.yfield` partitioned into `(Rx, Ry)` processes in ``(x, z)``.
7. Transpose + communicate to `storage.zfield` partitioned into `(Rx, Ry)` processes in ``(x, y)``.
8. Solve the tri-diagonal linear system in the ``z`` direction.

Steps 7 -> 1 are reversed to obtain the result in physical space stored in `storage.zfield` 
partitioned over ``(x, y)``.

Algorithm for slab decompositions
============================================

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
    - same as for two-dimensional decompositions with `Rx` (or `Ry`) equal to one

"""
function DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid, planner_flag=FFTW.PATIENT)
    
    validate_poisson_solver_distributed_grid(global_grid)
    validate_poisson_solver_configuration(global_grid, local_grid)
 
    irreg_dim = stretched_dimensions(local_grid)[1]

    topology(global_grid, irreg_dim) != Bounded && error("`DistributedFourierTridiagonalPoissonSolver` can only be used when the stretched direction's topology is `Bounded`.")

    FT         = Complex{eltype(local_grid)}
    child_arch = child_architecture(local_grid)
    storage    = TransposableField(CenterField(local_grid), FT)

    topo = (TX, TY, TZ) = topology(global_grid)
    λx = dropdims(poisson_eigenvalues(global_grid.Nx, global_grid.Lx, 1, TX()), dims=(2, 3))
    λy = dropdims(poisson_eigenvalues(global_grid.Ny, global_grid.Ly, 2, TY()), dims=(1, 3))
    λz = dropdims(poisson_eigenvalues(global_grid.Nz, global_grid.Lz, 3, TZ()), dims=(1, 2))
        
    if irreg_dim == 1
        arch = architecture(storage.xfield.grid)
        grid = storage.xfield.grid
        λ1 = partition_coordinate(λy, size(grid, 2), arch, 2)
        λ2 = partition_coordinate(λz, size(grid, 3), arch, 3)
    elseif irreg_dim == 2
        arch = architecture(storage.yfield.grid)
        grid = storage.yfield.grid
        λ1 = partition_coordinate(λx, size(grid, 1), arch, 1)
        λ2 = partition_coordinate(λz, size(grid, 3), arch, 3)
    elseif irreg_dim == 3
        arch = architecture(storage.zfield.grid)
        grid = storage.zfield.grid
        λ1 = partition_coordinate(λx, size(grid, 1), arch, 1)
        λ2 = partition_coordinate(λy, size(grid, 2), arch, 2)
    end

    λ1 = on_architecture(child_arch, λ1)
    λ2 = on_architecture(child_arch, λ2)

    plan = plan_distributed_transforms(global_grid, storage, planner_flag)

    # Lower and upper diagonals are the same
    lower_diagonal = @allowscalar [ 1 / Δξᶠ(q, grid) for q in 2:size(grid, irreg_dim) ]
    lower_diagonal = on_architecture(child_arch, lower_diagonal)
    upper_diagonal = lower_diagonal

    # Compute diagonal coefficients for each grid point
    diagonal = on_architecture(arch, zeros(size(grid)...))
    launch_config = if irreg_dim == 1
                        :yz
                    elseif irreg_dim == 2
                        :xz
                    elseif irreg_dim == 3
                        :xy
                    end

    tridiagonal_direction = stretched_direction(grid)
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

    buffer = if irreg_dim == 1
        (; y = buffer_y, z = buffer_z)
    elseif irreg_dim == 2
        (; x = buffer_x, z = buffer_z)
    elseif irreg_dim == 3
        (; x = buffer_x, y = buffer_y)
    end

    if irreg_dim == 1
        forward  = (y! = plan.forward.y!,  z! = plan.forward.z!)
        backward = (y! = plan.backward.y!, z! = plan.backward.z!)
    elseif irreg_dim == 2
        forward  = (x! = plan.forward.x!,  z! = plan.forward.z!)
        backward = (x! = plan.backward.x!, z! = plan.backward.z!)
    elseif irreg_dim == 3
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
function solve!(x, solver::DistributedFourierTridiagonalPoissonSolver{<:XYRegularRG})
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

function solve!(x, solver::DistributedFourierTridiagonalPoissonSolver{<:XZRegularRG})
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

function solve!(x, solver::DistributedFourierTridiagonalPoissonSolver{<:YZRegularRG})
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