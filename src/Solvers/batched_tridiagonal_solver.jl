using Oceananigans.Architectures: on_architecture
using Oceananigans.Grids: XDirection, YDirection, ZDirection

"""
    struct BatchedTridiagonalSolver{A, B, C, T, G, P, D, Β}

A batched solver for large numbers of triadiagonal systems.
"""
struct BatchedTridiagonalSolver{A, B, C, T, G, P, D, Β}
    a :: A
    b :: B
    c :: C
    t :: T
    grid :: G
    parameters :: P
    tridiagonal_direction :: D
    precomputed_pivots :: Β
end

function Base.summary(solver::BatchedTridiagonalSolver)
    dirstr = prettysummary(solver.tridiagonal_direction)
    return "BatchedTridiagonalSolver in $dirstr"
end

function Base.show(io::IO, solver::BatchedTridiagonalSolver)
    print(io, summary(solver), '\n')
    print(io, "└── grid: ", prettysummary(solver.grid))
end

# Some aliases...
const XTridiagonalSolver = BatchedTridiagonalSolver{A, B, C, T, G, P, <:XDirection} where {A, B, C, T, G, P}
const YTridiagonalSolver = BatchedTridiagonalSolver{A, B, C, T, G, P, <:YDirection} where {A, B, C, T, G, P}
const ZTridiagonalSolver = BatchedTridiagonalSolver{A, B, C, T, G, P, <:ZDirection} where {A, B, C, T, G, P}

Architectures.architecture(solver::BatchedTridiagonalSolver) = architecture(solver.grid)

"""
    BatchedTridiagonalSolver(grid;
                             lower_diagonal,
                             diagonal,
                             upper_diagonal,
                             scratch = zeros(architecture(grid), eltype(grid), grid.Nx, grid.Ny, grid.Nz),
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

Precomputed pivots
==================

The forward sweep computes the pivots

```
    βⁱʲ¹ = bⁱʲ¹,
    βⁱʲᵏ = bⁱʲᵏ - aⁱʲᵏ⁻¹ cⁱʲᵏ⁻¹ / βⁱʲᵏ⁻¹,  k = 2, ..., N
```

which depend only on the coefficients. Each pivot depends on the one before it, so this
recurrence sets the length of the sweep's dependency chain. When the same system is solved
many times with unchanged coefficients, the caller can compute the pivots once and pass the
array that holds them as `precomputed_pivots` (for example, a `CenterField(grid)` for a
`ZDirection` system of `Nz` levels). `solve!` then reads `β` from `precomputed_pivots`
instead of computing it, and does not evaluate the diagonal `b`.

The caller fills `precomputed_pivots`, indexed as `β[i, j, k]` over the interior, and
recomputes it whenever the coefficients change. Pivots computed with the recurrence above,
in the same floating-point operations as the sweep, give a solution bitwise identical to the
default, `precomputed_pivots = nothing`.
"""
function BatchedTridiagonalSolver(grid;
                                  lower_diagonal,
                                  diagonal,
                                  upper_diagonal,
                                  scratch = zeros(architecture(grid), eltype(grid), worksize(grid)...),
                                  parameters = nothing,
                                  tridiagonal_direction = ZDirection(),
                                  precomputed_pivots = nothing)

    return BatchedTridiagonalSolver(lower_diagonal, diagonal, upper_diagonal,
                                    scratch, grid, parameters, tridiagonal_direction,
                                    precomputed_pivots)
end

"""
$(TYPEDSIGNATURES)

Solve the batched tridiagonal system of linear equations with right hand side
`rhs` and lower diagonal, diagonal, and upper diagonal coefficients described by the
`BatchedTridiagonalSolver` `solver`. `BatchedTridiagonalSolver` uses a modified
TriDiagonal Matrix Algorithm (TDMA).

The result is stored in `ϕ` which must have size `(grid.Nx, grid.Ny, grid.Nz)`.

The `ZDirection` sweep masks products with a zero off-diagonal, so that `0 * NaN` gives `0`
rather than `NaN`. A `NaN` in an inactive cell of an immersed column therefore stays there
instead of spreading through the column. The `XDirection` and `YDirection` sweeps do not
mask, as they are not used on immersed grids.

Implementation follows [Press1992](@citet); §2.4. Note that a slightly different notation from
Press et al. is used for indexing the off-diagonal elements; see [`BatchedTridiagonalSolver`](@ref).

Reference
=========

Press William, H., Teukolsky Saul, A., Vetterling William, T., & Flannery Brian, P. (1992).
    Numerical recipes: the art of scientific computing. Cambridge University Press
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
            solver.tridiagonal_direction,
            solver.precomputed_pivots)

    return nothing
end

@inline get_coefficient(i, j, k, grid, a::AbstractArray{<:Any, 1}, p, ::XDirection,          args...) = @inbounds a[i]
@inline get_coefficient(i, j, k, grid, a::AbstractArray{<:Any, 1}, p, ::YDirection,          args...) = @inbounds a[j]
@inline get_coefficient(i, j, k, grid, a::AbstractArray{<:Any, 1}, p, ::ZDirection,          args...) = @inbounds a[k]
@inline get_coefficient(i, j, k, grid, a::AbstractArray{<:Any, 3}, p, tridiagonal_direction, args...) = @inbounds a[i, j, k]

# `a * b` if `a != 0`, 0 otherwise (masks away NaNs in `b`)
@inline masked_multiply(x, y) = x * y * (x != 0)

@inline float_eltype(ϕ::AbstractArray{T}) where T <: AbstractFloat = T
@inline float_eltype(ϕ::AbstractArray{<:Complex{T}}) where T <: AbstractFloat = T

# The `ZDirection` sweep masks products with a zero off-diagonal (see `solve!`)
@inline off_diagonal_multiply(::ZDirection, x, y) = masked_multiply(x, y)
@inline off_diagonal_multiply(tridiagonal_direction, x, y) = x * y

# Index `(i, j, k)` of level `n` of the system in column `(m₁, m₂)`
@inline column_index(n, m₁, m₂, ::XDirection) = (n, m₁, m₂)
@inline column_index(n, m₁, m₂, ::YDirection) = (m₁, n, m₂)
@inline column_index(n, m₁, m₂, ::ZDirection) = (m₁, m₂, n)

# Pivots are either computed by the forward sweep or read from `precomputed_pivots`
@inline first_pivot(i, j, k, grid, b, p, tridiagonal_direction, args, ::Nothing) =
    get_coefficient(i, j, k, grid, b, p, tridiagonal_direction, args...)

@inline first_pivot(i, j, k, grid, b, p, tridiagonal_direction, args, β) = @inbounds β[i, j, k]

@inline next_pivot(i, j, k, grid, b, p, tridiagonal_direction, args, ::Nothing, aᵏ⁻¹, tᵏ) =
    get_coefficient(i, j, k, grid, b, p, tridiagonal_direction, args...) -
    off_diagonal_multiply(tridiagonal_direction, aᵏ⁻¹, tᵏ)

@inline next_pivot(i, j, k, grid, b, p, tridiagonal_direction, args, β, aᵏ⁻¹, tᵏ) = @inbounds β[i, j, k]

@kernel function solve_batched_tridiagonal_system_kernel!(ϕ, a, b, c, f, t, grid, p, args,
                                                          tridiagonal_direction::XDirection, pivots)
    j, k = @index(Global, NTuple)
    solve_batched_tridiagonal_system!(j, k, size(grid, 1), ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction, pivots)
end

@kernel function solve_batched_tridiagonal_system_kernel!(ϕ, a, b, c, f, t, grid, p, args,
                                                          tridiagonal_direction::YDirection, pivots)
    i, k = @index(Global, NTuple)
    solve_batched_tridiagonal_system!(i, k, size(grid, 2), ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction, pivots)
end

@kernel function solve_batched_tridiagonal_system_kernel!(ϕ, a, b, c, f, t, grid, p, args,
                                                          tridiagonal_direction::ZDirection, pivots)
    i, j = @index(Global, NTuple)
    solve_batched_tridiagonal_system!(i, j, size(grid, 3), ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction, pivots)
end

# Thomas sweep along the column `(m₁, m₂)`, which has `N` levels.
#
# Level `n` depends on level `n ∓ 1`, and each column is swept by a single thread, so the sweep's
# run time is the length of that dependency chain. The pivot `β`, the elimination ratio `tᵏ`, and
# the solution `ϕ⁻` / `ϕ⁺` of the level just solved are therefore carried in local variables,
# rather than read back from the arrays they were just stored in: the compiler cannot prove that
# `ϕ` and `t` do not alias the coefficients, so it cannot forward a stored value to the next
# load by itself. Each carried value is converted to the element type of the array it is stored
# in, so it equals the value a re-read would return and the result is unchanged.
#
# Coefficients of level `n`, and its right-hand side, are evaluated before `ϕ` is written at
# level `n`, so in-place solves (`rhs === ϕ`) and coefficients that read `ϕ` see the same values
# as they would without the carried variables.
@inline function solve_batched_tridiagonal_system!(m₁, m₂, N, ϕ, a, b, c, f, t, grid, p, args,
                                                   tridiagonal_direction, pivots)
    Φ = eltype(ϕ)
    Τ = eltype(t)

    @inbounds begin
        i, j, k = column_index(1, m₁, m₂, tridiagonal_direction)
        β  = first_pivot(i, j, k, grid, b, p, tridiagonal_direction, args, pivots)
        f₁ = get_coefficient(i, j, k, grid, f, p, tridiagonal_direction, args...)
        ϕ⁻ = convert(Φ, f₁ / β)
        ϕ[i, j, k] = ϕ⁻

        for n = 2:N
            i⁻, j⁻, k⁻ = column_index(n-1, m₁, m₂, tridiagonal_direction)
            i,  j,  k  = column_index(n,   m₁, m₂, tridiagonal_direction)

            cᵏ⁻¹ = get_coefficient(i⁻, j⁻, k⁻, grid, c, p, tridiagonal_direction, args...)
            aᵏ⁻¹ = get_coefficient(i⁻, j⁻, k⁻, grid, a, p, tridiagonal_direction, args...)

            tᵏ = convert(Τ, cᵏ⁻¹ / β)
            t[i, j, k] = tᵏ
            β = next_pivot(i, j, k, grid, b, p, tridiagonal_direction, args, pivots, aᵏ⁻¹, tᵏ)

            fᵏ = get_coefficient(i, j, k, grid, f, p, tridiagonal_direction, args...)

            # If the problem is not diagonally-dominant such that `β ≈ 0`,
            # the algorithm is unstable and we store the right-hand side instead.
            definitely_diagonally_dominant = abs(β) > 10 * eps(float_eltype(ϕ))
            ϕ★ = (fᵏ - off_diagonal_multiply(tridiagonal_direction, aᵏ⁻¹, ϕ⁻)) / β
            ϕ⁻ = convert(Φ, ifelse(definitely_diagonally_dominant, ϕ★, fᵏ))
            ϕ[i, j, k] = ϕ⁻
        end

        ϕ⁺ = ϕ⁻
        for n = N-1:-1:1
            i,  j,  k  = column_index(n,   m₁, m₂, tridiagonal_direction)
            i⁺, j⁺, k⁺ = column_index(n+1, m₁, m₂, tridiagonal_direction)
            ϕ⁺ = convert(Φ, ϕ[i, j, k] - off_diagonal_multiply(tridiagonal_direction, t[i⁺, j⁺, k⁺], ϕ⁺))
            ϕ[i, j, k] = ϕ⁺
        end
    end
end
