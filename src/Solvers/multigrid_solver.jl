using Oceananigans.Architectures: architecture
using Oceananigans.Grids: interior_parent_indices
using Statistics: norm, dot
using LinearAlgebra
using AlgebraicMultigrid: _solve!, init, RugeStubenAMG
using Oceananigans.Operators: volume, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶜᶜᵃ, Δyᵃᶜᵃ, Δxᶜᵃᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, ∇²ᶜᶜᶜ

import Oceananigans.Architectures: architecture

mutable struct MultigridSolver{A, G, L, T, M, F}
               architecture :: A
                       grid :: G
            linear_operator :: L
                  tolerance :: T
         maximum_iterations :: Int
              amg_algorithm :: M
                    x_array :: F
                    b_array :: F
end

architecture(solver::MultigridSolver) = solver.architecture
    
"""
    MultigridSolver(grid,
                    linear_operation!, 
                    args...;
                    maximum_iterations = 100,
                    tolerance = 1e-13,
                    amg_algorithm = RugeStubenAMG(),
                    )

Returns a MultigridSolver that solves the linear equation
``A x = b`` using a multigrid method, where `A * x` is
determined by `linear_operation!`

`linear_operation!` is a function with signature `linear_operation!(Ax, x, args...)` 
that calculates `A * x` for given `x` and stores the result in `Ax`.


The solver is used by calling

```
solve!(x, solver::MultigridSolver, b; kwargs...)
```

for `solver`, right-hand side `b`, solution `x`, and optional keyword arguments `kwargs...`.

Arguments
=========

* `maximum_iterations`: Maximum number of iterations the solver may perform before exiting.

* `tolerance`: Tolerance for convergence of the algorithm. The algorithm quits when
               `norm(A * x - b) < tolerance`.
"""
function MultigridSolver(linear_operation!,
                         args...;
                         template_field::AbstractField,
                         maximum_iterations = 100, #prod(size(template_field)),
                         tolerance = 1e-13, #sqrt(eps(eltype(template_field.grid))),
                         amg_algorithm = RugeStubenAMG(),
                         )

    arch = architecture(template_field)
    grid = template_field.grid

    matrix = create_matrix(grid, linear_operation!, args...)


    Nx, Ny, Nz = size(grid)

    _, _, LZ = location(template_field)

    (LZ == Nothing) && (Nz = 1)

    b_array = arch_array(arch, zeros(Nx * Ny * Nz))
    x_array = arch_array(arch, zeros(Nx * Ny * Nz))

    return MultigridSolver(arch,
                           grid,
                           matrix,
                           tolerance,
                           maximum_iterations,
                           amg_algorithm,
                           x_array,
                           b_array
                           )
end

# TODO make inplace create_matrix!

# For free surface without Δt
function create_matrix(grid, linear_operator!, ::Any , ::Any, ::Any, ::Nothing)
    Nx, Ny, _ = size(grid)
    return spzeros(eltype(grid), Nx*Ny, Nx*Ny)
end

function create_matrix(grid, linear_operator!, args...)
    Nx, Ny, _ = size(grid)
    A = spzeros(eltype(grid), Nx*Ny, Nx*Ny)

    create_matrix!(A, grid, linear_operator!, args...)
    
    return A
end

function create_matrix!(A, grid, linear_operator!, args...)
    Nx, Ny, _ = size(grid)
    make_column(f) = reshape(interior(f), (Nx*Ny, 1))

    eᵢⱼ = Field{Center, Center, Nothing}(grid)
    ∇²eᵢⱼ = Field{Center, Center, Nothing}(grid)
    
    for j in 1:Ny, i in 1:Nx
        eᵢⱼ .= 0
        ∇²eᵢⱼ .= 0
        eᵢⱼ[i, j] = 1
        fill_halo_regions!(eᵢⱼ)
        linear_operator!(∇²eᵢⱼ, eᵢⱼ, args...)

        A[:, Nx*(j-1) + i] .= make_column(∇²eᵢⱼ)
    end
end

"""
    solve!(x, solver::MultigridSolver, b; kwargs...)

Solve `A * x = b` using a multigrid method, where `A * x` is
determined by `linear_operation!` given in the MultigridSolver constructor.
"""
function solve!(x, solver::MultigridSolver, b; kwargs...)
    grid = b.grid
    Nx, Ny, Nz = size(grid)

    solver.b_array .= reshape(interior(b), Nx * Ny * Nz)
    solver.x_array .= reshape(interior(x), Nx * Ny * Nz)

    solt = init(solver.amg_algorithm, solver.linear_operator, solver.b_array)

    _solve!(solver.x_array, solt.ml, solt.b, maxiter=solver.maximum_iterations, abstol = solver.tolerance, kwargs...)
    
    interior(x) .= reshape(solver.x_array, Nx, Ny, Nz)
    fill_halo_regions!(x)
end

function solve!(x::Field{Center, Center, Nothing}, solver::MultigridSolver, b::Field{Center, Center, Nothing}; kwargs...)
    grid = b.grid
    Nx, Ny, _ = size(grid)

    solver.b_array .= reshape(interior(b), Nx * Ny)
    solver.x_array .= reshape(interior(x), Nx * Ny)

    solt = init(solver.amg_algorithm, solver.linear_operator, solver.b_array)

    _solve!(solver.x_array, solt.ml, solt.b, maxiter=solver.maximum_iterations, abstol = solver.tolerance, kwargs...)
    
    interior(x) .= reshape(solver.x_array, Nx, Ny)
    fill_halo_regions!(x)
end

function Base.show(io::IO, solver::MultigridSolver)
    print(io, "Multigrid solver.\n")
    print(io, " Problem size = "  , size(solver.grid), '\n')
    print(io, " Grid = "  , solver.grid)
    return nothing
end
