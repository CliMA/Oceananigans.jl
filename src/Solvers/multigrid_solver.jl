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
                     abstol :: T
                     reltol :: T
                    maxiter :: Int
              amg_algorithm :: M
                    x_array :: F
                    b_array :: F
end

architecture(solver::MultigridSolver) = solver.architecture
    
"""
    MultigridSolver(linear_operation!::Function,
                    args...;
                    template_field::AbstractField,
                    maxiter = prod(size(template_field)),
                    reltol = sqrt(eps(eltype(template_field.grid))),
                    abstol = 0reltol,
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

* `template_field`: Dummy field that is the same type and size as `x` and `b`, which
                    is used to infer the `architecture`, `grid`, and to create work arrays
                    that are used internally by the solver.

* `maxiter`: Maximum number of iterations the solver may perform before exiting.

* `reltol, abstol`: Relative and absolute tolerance for convergence of the algorithm.
                    The iteration stops when `norm(A * x - b) < tolerance`.

* `amg_algorithm`: Algebraic Multigrid algorithm defining mapping between different grid spacings

!!! compat "Multigrid solver on GPUs"
    Currently Multigrid solver is only supported on CPUs.
"""
function MultigridSolver(linear_operation!::Function,
                         args...;
                         template_field::AbstractField,
                         maxiter = prod(size(template_field)),
                         reltol = sqrt(eps(eltype(template_field.grid))),
                         abstol = 0,
                         amg_algorithm = RugeStubenAMG(),
                         )

    arch = architecture(template_field)

    arch == GPU() && error("Multigrid solver is only supported on CPUs.")

    matrix = initialize_matrix(template_field, linear_operation!, args...)

    Nx, Ny, Nz = size(template_field)

    FT = eltype(template_field.grid)

    b_array = arch_array(arch, zeros(FT, Nx * Ny * Nz))
    x_array = arch_array(arch, zeros(FT, Nx * Ny * Nz))

    return MultigridSolver(arch,
                           template_field.grid,
                           matrix,
                           FT(abstol),
                           FT(reltol),
                           maxiter,
                           amg_algorithm,
                           x_array,
                           b_array
                           )
end


# For free surface without Δt
function initialize_matrix(template_field, ::Function, ::Any , ::Any, ::Any, ::Nothing)
    Nx, Ny, Nz = size(template_field)
    return spzeros(eltype(template_field.grid), Nx*Ny*Nz, Nx*Ny*Nz)
end

function initialize_matrix(template_field, linear_operator!, args...)
    Nx, Ny, Nz = size(template_field)
    A = spzeros(eltype(template_field.grid), Nx*Ny*Nz, Nx*Ny*Nz)

    fill_matrix_elements!(A, template_field, linear_operator!, args...)
    
    return A
end

function fill_matrix_elements!(A, template_field, linear_operator!, args...)
    Nx, Ny, Nz = size(template_field)
    make_column(f) = reshape(interior(f), Nx*Ny*Nz)

    eᵢⱼₖ = similar(template_field)
    ∇²eᵢⱼₖ = similar(template_field)
    
    for k = 1:Nz, j in 1:Ny, i in 1:Nx
        eᵢⱼₖ .= 0
        ∇²eᵢⱼₖ .= 0
        eᵢⱼₖ[i, j, k] = 1
        fill_halo_regions!(eᵢⱼₖ)
        linear_operator!(∇²eᵢⱼₖ, eᵢⱼₖ, args...)

        A[:, Ny*Nx*(k-1) + Nx*(j-1) + i] .= make_column(∇²eᵢⱼₖ)
    end
    
    return nothing
end

"""
    solve!(x, solver::MultigridSolver, b; kwargs...)

Solve `A * x = b` using a multigrid method, where `A * x` is
determined by `linear_operation!` given in the MultigridSolver constructor.
"""
function solve!(x, solver::MultigridSolver, b; kwargs...)
    Nx, Ny, Nz = size(b)

    solver.b_array .= reshape(interior(b), Nx * Ny * Nz)
    solver.x_array .= reshape(interior(x), Nx * Ny * Nz)

    solt = init(solver.amg_algorithm, solver.linear_operator, solver.b_array)

    _solve!(solver.x_array, solt.ml, solt.b, maxiter=solver.maxiter, abstol = solver.abstol, reltol=solver.reltol, kwargs...)
    
    interior(x) .= reshape(solver.x_array, Nx, Ny, Nz)
    fill_halo_regions!(x)
end


function Base.show(io::IO, solver::MultigridSolver)
    print(io, "Multigrid solver.\n")
    print(io, " Problem size = "  , size(solver.grid), '\n')
    print(io, " Grid = "  , solver.grid)
    return nothing
end
