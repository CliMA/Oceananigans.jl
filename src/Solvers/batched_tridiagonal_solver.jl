using Oceananigans.Architectures: arch_array
import Oceananigans.Architectures: architecture

"""
    BatchedTridiagonalSolver

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
    BatchedTridiagonalSolver(grid; lower_diagonal, diagonal, upper_diagonal, parameters=nothing)

Construct a solver for batched tridiagonal systems on `grid` of the form

                           bⁱʲ¹ ϕⁱʲ¹ + cⁱʲ¹ ϕⁱʲ²   = fⁱʲ¹,  k = 1
           aⁱʲᵏ⁻¹ ϕⁱʲᵏ⁻¹ + bⁱʲᵏ ϕⁱʲᵏ + cⁱʲᵏ ϕⁱʲᵏ⁺¹ = fⁱʲᵏ,  k = 2, ..., N-1
           aⁱʲᴺ⁻¹ ϕⁱʲᴺ⁻¹ + bⁱʲᴺ ϕⁱʲᴺ               = fⁱʲᴺ,  k = N

where `a` is the `lower_diagonal`, `b` is the `diagonal`, and `c` is the `upper_diagonal`.
`ϕ` is the solution and `f` is the right hand side source term passed to `solve!(ϕ, tridiagonal_solver, f)`

`a`, `b`, `c`, and `f` can be specified in three ways:

1. A 1D array means that `aⁱʲᵏ = a[k]`.

2. A 3D array means that `aⁱʲᵏ = a[i, j, k]`.

3. Otherwise, `a` is assumed to be callable:
    * If `isnothing(parameters)` then `aⁱʲᵏ = a(i, j, k, grid, args...)`.
    * If `!isnothing(parameters)` then `aⁱʲᵏ = a(i, j, k, grid, parameters, args...)`.
    where `args...` are `Varargs` passed to `solve_batched_tridiagonal_system!(ϕ, solver, args...)`.
"""
function BatchedTridiagonalSolver(grid;
                                  lower_diagonal,
                                  diagonal,
                                  upper_diagonal,
                                  scratch = arch_array(architecture(grid), zeros(eltype(grid), grid.Nx, grid.Ny, grid.Nz)),
                                  parameters = nothing,
                                  tridiagonal_direction = :z)

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

Reference implementation per Numerical Recipes, Press et. al 1992 (§ 2.4).
"""
function solve!(ϕ, solver::BatchedTridiagonalSolver, rhs, args... )

    launch_config = if solver.tridiagonal_direction == :x
                        :yz
                    elseif solver.tridiagonal_direction == :y
                        :xz
                    elseif solver.tridiagonal_direction == :z
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
            Val(solver.tridiagonal_direction))

    return nothing
end

@inline get_coefficient(a::AbstractArray{T, 1}, i, j, k, grid, p, ::Val{:x},   args...) where {T} = @inbounds a[i]
@inline get_coefficient(a::AbstractArray{T, 1}, i, j, k, grid, p, ::Val{:y},   args...) where {T} = @inbounds a[j]
@inline get_coefficient(a::AbstractArray{T, 1}, i, j, k, grid, p, ::Val{:z},   args...) where {T} = @inbounds a[k]
@inline get_coefficient(a::AbstractArray{T, 3}, i, j, k, grid, p, tridiag_dir, args...) where {T} = @inbounds a[i, j, k]

@inline get_coefficient(a::Base.Callable, i, j, k, grid, p,         tridiagonal_direction, args...) = a(i, j, k, grid, p, args...)
@inline get_coefficient(a::Base.Callable, i, j, k, grid, ::Nothing, tridiagonal_direction, args...) = a(i, j, k, grid, args...)

@inline float_eltype(ϕ::AbstractArray{T}) where T <: AbstractFloat = T
@inline float_eltype(ϕ::AbstractArray{<:Complex{T}}) where T <: AbstractFloat = T

@kernel function solve_batched_tridiagonal_system_kernel!(ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction::Val{:x})
    N = size(grid, 1)
    m, n = @index(Global, NTuple)

    @inbounds begin
        β  = get_coefficient(b, 1, m, n, grid, p, tridiagonal_direction, args...)
        f₁ = get_coefficient(f, 1, m, n, grid, p, tridiagonal_direction, args...)
        ϕ[1, m, n] = f₁ / β

        @unroll for q = 2:N
            cᵏ⁻¹ = get_coefficient(c, q-1, m, n, grid, p, tridiagonal_direction, args...)
            bᵏ   = get_coefficient(b, q,   m, n, grid, p, tridiagonal_direction, args...)
            aᵏ⁻¹ = get_coefficient(a, q-1, m, n, grid, p, tridiagonal_direction, args...)

            t[q, m, n] = cᵏ⁻¹ / β
            β = bᵏ - aᵏ⁻¹ * t[q, m, n]

            fᵏ = get_coefficient(f, q, m, n, grid, p, tridiagonal_direction, args...)

            # If the problem is not diagonally-dominant such that `β ≈ 0`,
            # the algorithm is unstable and we elide the forward pass update of ϕ.
            definitely_diagonally_dominant = abs(β) > 10 * eps(float_eltype(ϕ))
            !definitely_diagonally_dominant && break
            ϕ[q, m, n] = (fᵏ - aᵏ⁻¹ * ϕ[q-1, m, n]) / β
        end

        @unroll for q = N-1:-1:1
            ϕ[q, m, n] -= t[q+1, m, n] * ϕ[q+1, m, n]
        end
    end
end

@kernel function solve_batched_tridiagonal_system_kernel!(ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction::Val{:y})
    N = size(grid, 2)
    m, n = @index(Global, NTuple)

    @inbounds begin
        β  = get_coefficient(b, m, 1, n, grid, p, tridiagonal_direction, args...)
        f₁ = get_coefficient(f, m, 1, n, grid, p, tridiagonal_direction, args...)
        ϕ[m, 1, n] = f₁ / β

        @unroll for q = 2:N
            cᵏ⁻¹ = get_coefficient(c, m, q-1, n, grid, p, tridiagonal_direction, args...)
            bᵏ   = get_coefficient(b, m, q,   n, grid, p, tridiagonal_direction, args...)
            aᵏ⁻¹ = get_coefficient(a, m, q-1, n, grid, p, tridiagonal_direction, args...)

            t[m, q, n] = cᵏ⁻¹ / β
            β = bᵏ - aᵏ⁻¹ * t[m, q, n]

            fᵏ = get_coefficient(f, m, q, n, grid, p, tridiagonal_direction, args...)

            # If the problem is not diagonally-dominant such that `β ≈ 0`,
            # the algorithm is unstable and we elide the forward pass update of ϕ.
            definitely_diagonally_dominant = abs(β) > 10 * eps(float_eltype(ϕ))
            !definitely_diagonally_dominant && break
            ϕ[m, q, n] = (fᵏ - aᵏ⁻¹ * ϕ[m, q-1, n]) / β
        end

        @unroll for q = N-1:-1:1
            ϕ[m, q, n] -= t[m, q+1, n] * ϕ[m, q+1, n]
        end
    end
end

@kernel function solve_batched_tridiagonal_system_kernel!(ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction::Val{:z})
    N = size(grid, 3)
    m, n = @index(Global, NTuple)

    @inbounds begin
        β  = get_coefficient(b, m, n, 1, grid, p, tridiagonal_direction, args...)
        f₁ = get_coefficient(f, m, n, 1, grid, p, tridiagonal_direction, args...)
        ϕ[m, n, 1] = f₁ / β

        @unroll for q = 2:N
            cᵏ⁻¹ = get_coefficient(c, m, n, q-1, grid, p, tridiagonal_direction, args...)
            bᵏ   = get_coefficient(b, m, n, q,   grid, p, tridiagonal_direction, args...)
            aᵏ⁻¹ = get_coefficient(a, m, n, q-1, grid, p, tridiagonal_direction, args...)

            t[m, n, q] = cᵏ⁻¹ / β
            β = bᵏ - aᵏ⁻¹ * t[m, n, q]

            fᵏ = get_coefficient(f, m, n, q, grid, p, tridiagonal_direction, args...)

            # If the problem is not diagonally-dominant such that `β ≈ 0`,
            # the algorithm is unstable and we elide the forward pass update of ϕ.
            definitely_diagonally_dominant = abs(β) > 10 * eps(float_eltype(ϕ))
            !definitely_diagonally_dominant && break
            ϕ[m, n, q] = (fᵏ - aᵏ⁻¹ * ϕ[m, n, q-1]) / β
        end

        @unroll for q = N-1:-1:1
            ϕ[m, n, q] -= t[m, n, q+1] * ϕ[m, n, q+1]
        end
    end
end
