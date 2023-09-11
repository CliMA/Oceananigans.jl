using Oceananigans.Architectures: arch_array
using Oceananigans.Grids: XDirection, YDirection, ZDirection

import Oceananigans.Architectures: architecture

"""
    struct BatchedTridiagonalSolver{A, B, C, T, G, P}

A batched solver for large numbers of triadiagonal systems.
"""
struct BatchedTridiagonalSolver{A, B, C, T, G, P, D}
    a :: A
    b :: B
    c :: C
    t :: T
    grid :: G
    parameters :: P
    tridiagonal_direction :: D
end

architecture(solver::BatchedTridiagonalSolver) = architecture(solver.grid)


"""
    BatchedTridiagonalSolver(grid;
                             lower_diagonal,
                             diagonal,
                             upper_diagonal,
                             scratch = arch_array(architecture(grid), zeros(eltype(grid), size(grid)...)),
                             tridiagonal_direction = ZDirection()
                             parameters = nothing)

Construct a solver for batched tridiagonal systems on `grid` of the form

```
                    bⁱʲ¹ ϕⁱʲ¹ + cⁱʲ¹ ϕⁱʲ²   = fⁱʲ¹,
    aⁱʲᵏ⁻¹ ϕⁱʲᵏ⁻¹ + bⁱʲᵏ ϕⁱʲᵏ + cⁱʲᵏ ϕⁱʲᵏ⁺¹ = fⁱʲᵏ,  k = 2, ..., N-1
    aⁱʲᴺ⁻¹ ϕⁱʲᴺ⁻¹ + bⁱʲᴺ ϕⁱʲᴺ               = fⁱʲᴺ,
```
or in matrix form
```
    ⎡ bⁱʲ¹   cⁱʲ¹     0       ⋯         0   ⎤ ⎡ ϕⁱʲ¹ ⎤   ⎡ fⁱʲ¹ ⎤
    ⎢ aⁱʲ¹   bⁱʲ²   cⁱʲ²      0    ⋯    ⋮   ⎥ ⎢ ϕⁱʲ² ⎥   ⎢ fⁱʲ² ⎥
    ⎢  0      ⋱      ⋱       ⋱              ⎥ ⎢   .  ⎥   ⎢   .  ⎥
    ⎢  ⋮                                0   ⎥ ⎢ ϕⁱʲᵏ ⎥   ⎢ fⁱʲᵏ ⎥
    ⎢  ⋮           aⁱʲᴺ⁻²   bⁱʲᴺ⁻¹   cⁱʲᴺ⁻¹ ⎥ ⎢      ⎥   ⎢   .  ⎥
    ⎣  0      ⋯      0      aⁱʲᴺ⁻¹    bⁱʲᴺ  ⎦ ⎣ ϕⁱʲᴺ ⎦   ⎣ fⁱʲᴺ ⎦
```

where `a` is the `lower_diagonal`, `b` is the `diagonal`, and `c` is the `upper_diagonal`.

Note the convention used here for indexing the upper and lower diagonals; this can be different from 
other implementations where, e.g., `aⁱʲ²` may appear at the second row, instead of `aⁱʲ¹` as above.

`ϕ` is the solution and `f` is the right hand side source term passed to `solve!(ϕ, tridiagonal_solver, f)`.

`a`, `b`, `c`, and `f` can be specified in three ways:

1. A 1D array means, e.g., that `aⁱʲᵏ = a[k]`.

2. A 3D array means, e.g., that `aⁱʲᵏ = a[i, j, k]`.

Other coefficient types can be implemented by extending `get_coefficient`.
"""
function BatchedTridiagonalSolver(grid;
                                  lower_diagonal,
                                  diagonal,
                                  upper_diagonal,
                                  scratch = arch_array(architecture(grid), zeros(eltype(grid), grid.Nx, grid.Ny, grid.Nz)),
                                  parameters = nothing,
                                  tridiagonal_direction = ZDirection())

    return BatchedTridiagonalSolver(lower_diagonal, diagonal, upper_diagonal,
                                    scratch, grid, parameters, tridiagonal_direction)
end

"""
    solve!(ϕ, solver::BatchedTridiagonalSolver, rhs, args...)

Solve the batched tridiagonal system of linear equations with right hand side
`rhs` and lower diagonal, diagonal, and upper diagonal coefficients described by the
`BatchedTridiagonalSolver` `solver`. `BatchedTridiagonalSolver` uses a modified
TriDiagonal Matrix Algorithm (TDMA).

The result is stored in `ϕ` which must have size `(grid.Nx, grid.Ny, grid.Nz)`.

Reference implementation per Numerical Recipes, Press et al. 1992 (§ 2.4). Note that
a slightly different notation from Press et al. is used for indexing the off-diagonal
elements; see [`BatchedTridiagonalSolver`](@ref).
"""
function solve!(ϕ, solver::BatchedTridiagonalSolver, rhs, args...)

    launch_config = if solver.tridiagonal_direction isa XDirection
                        :yz
                    elseif solver.tridiagonal_direction isa YDirection
                        :xz
                    elseif solver.tridiagonal_direction isa ZDirection
                        :xy
                    end

    launch!(architecture(solver), solver.grid, launch_config,
            solve_batched_tridiagonal_system_kernel!, ϕ,
            solver.a,
            solver.b,
            solver.c,
            rhs,
            solver.t,
            solver.grid,
            solver.parameters,
            Tuple(args),
            solver.tridiagonal_direction)

    return nothing
end

@inline get_coefficient(i, j, k, grid, a::AbstractArray{<:Any, 1}, p, ::XDirection,          args...) = @inbounds a[i]
@inline get_coefficient(i, j, k, grid, a::AbstractArray{<:Any, 1}, p, ::YDirection,          args...) = @inbounds a[j]
@inline get_coefficient(i, j, k, grid, a::AbstractArray{<:Any, 1}, p, ::ZDirection,          args...) = @inbounds a[k]
@inline get_coefficient(i, j, k, grid, a::AbstractArray{<:Any, 3}, p, tridiagonal_direction, args...) = @inbounds a[i, j, k]

@inline float_eltype(ϕ::AbstractArray{T}) where T <: AbstractFloat = T
@inline float_eltype(ϕ::AbstractArray{<:Complex{T}}) where T <: AbstractFloat = T

@kernel function solve_batched_tridiagonal_system_kernel!(ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction::XDirection)
    Nx = size(grid, 1)
    j, k = @index(Global, NTuple)

    @inbounds begin
        β  = get_coefficient(1, j, k, grid, b, p, tridiagonal_direction, args...)
        f₁ = get_coefficient(1, j, k, grid, f, p, tridiagonal_direction, args...)
        ϕ[1, j, k] = f₁ / β

        @unroll for i = 2:Nx
            cᵏ⁻¹ = get_coefficient(i-1, j, k, grid, c, p, tridiagonal_direction, args...)
            bᵏ   = get_coefficient(i,   j, k, grid, b, p, tridiagonal_direction, args...)
            aᵏ⁻¹ = get_coefficient(i-1, j, k, grid, a, p, tridiagonal_direction, args...)

            t[i, j, k] = cᵏ⁻¹ / β
            β = bᵏ - aᵏ⁻¹ * t[i, j, k]

            fᵏ = get_coefficient(i, j, k, grid, f, p, tridiagonal_direction, args...)

            # If the problem is not diagonally-dominant such that `β ≈ 0`,
            # the algorithm is unstable and we elide the forward pass update of ϕ.
            definitely_diagonally_dominant = abs(β) > 10 * eps(float_eltype(ϕ))
            !definitely_diagonally_dominant && break
            ϕ[i, j, k] = (fᵏ - aᵏ⁻¹ * ϕ[i-1, j, k]) / β
        end

        @unroll for i = Nx-1:-1:1
            ϕ[i, j, k] -= t[i+1, j, k] * ϕ[i+1, j, k]
        end
    end
end

@kernel function solve_batched_tridiagonal_system_kernel!(ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction::YDirection)
    Ny = size(grid, 2)
    i, k = @index(Global, NTuple)

    @inbounds begin
        β  = get_coefficient(i, 1, k, grid, b, p, tridiagonal_direction, args...)
        f₁ = get_coefficient(i, 1, k, grid, f, p, tridiagonal_direction, args...)
        ϕ[i, 1, k] = f₁ / β

        @unroll for j = 2:Ny
            cᵏ⁻¹ = get_coefficient(i, j-1, k, grid, c, p, tridiagonal_direction, args...)
            bᵏ   = get_coefficient(i, j,   k, grid, b, p, tridiagonal_direction, args...)
            aᵏ⁻¹ = get_coefficient(i, j-1, k, grid, a, p, tridiagonal_direction, args...)

            t[i, j, k] = cᵏ⁻¹ / β
            β = bᵏ - aᵏ⁻¹ * t[i, j, k]

            fᵏ = get_coefficient(i, j, k, grid, f, p, tridiagonal_direction, args...)

            # If the problem is not diagonally-dominant such that `β ≈ 0`,
            # the algorithm is unstable and we elide the forward pass update of ϕ.
            definitely_diagonally_dominant = abs(β) > 10 * eps(float_eltype(ϕ))
            !definitely_diagonally_dominant && break
            ϕ[i, j, k] = (fᵏ - aᵏ⁻¹ * ϕ[i, j-1, k]) / β
        end

        @unroll for j = Ny-1:-1:1
            ϕ[i, j, k] -= t[i, j+1, k] * ϕ[i, j+1, k]
        end
    end
end

@kernel function solve_batched_tridiagonal_system_kernel!(ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction::ZDirection)
    Nz = size(grid, 3)
    i, j = @index(Global, NTuple)

    @inbounds begin
        β  = get_coefficient(i, j, 1, grid, b, p, tridiagonal_direction, args...)
        f₁ = get_coefficient(i, j, 1, grid, f, p, tridiagonal_direction, args...)
        ϕ[i, j, 1] = f₁ / β

        @unroll for k = 2:Nz
            cᵏ⁻¹ = get_coefficient(i, j, k-1, grid, c, p, tridiagonal_direction, args...)
            bᵏ   = get_coefficient(i, j, k,   grid, b, p, tridiagonal_direction, args...)
            aᵏ⁻¹ = get_coefficient(i, j, k-1, grid, a, p, tridiagonal_direction, args...)

            t[i, j, k] = cᵏ⁻¹ / β
            β = bᵏ - aᵏ⁻¹ * t[i, j, k]
            fᵏ = get_coefficient(i, j, k, grid, f, p, tridiagonal_direction, args...)

            # If the problem is not diagonally-dominant such that `β ≈ 0`,
            # the algorithm is unstable and we elide the forward pass update of `ϕ`.
            definitely_diagonally_dominant = abs(β) > 10 * eps(float_eltype(ϕ))
            !definitely_diagonally_dominant && break
            ϕ[i, j, k] = (fᵏ - aᵏ⁻¹ * ϕ[i, j, k-1]) / β
        end

        @unroll for k = Nz-1:-1:1
            ϕ[i, j, k] -= t[i, j, k+1] * ϕ[i, j, k+1]
        end
    end
end
