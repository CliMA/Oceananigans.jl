using Oceananigans.Architectures: on_architecture
using Oceananigans.Grids: XDirection, YDirection, ZDirection

"""
    struct BatchedTridiagonalSolver{A, B, C, T, G, P, D, F}

A batched solver for large numbers of triadiagonal systems.
"""
struct BatchedTridiagonalSolver{A, B, C, T, G, P, D, F}
    a :: A
    b :: B
    c :: C
    t :: T
    grid :: G
    parameters :: P
    tridiagonal_direction :: D
    factors :: F
end

"""
    TridiagonalFactors

The stored factorization of the systems of a factorized [`BatchedTridiagonalSolver`](@ref):
the pivots `β` and the lower diagonal `a`, evaluated once by [`compute_factorization!`](@ref).
The elimination ratios `t` are kept in the solver's scratch array. Together these are the
factors `A = L U` of each system, where `L` has diagonal `β` and sub-diagonal `a`, and `U` is unit
upper triangular with super-diagonal `t`.
"""
struct TridiagonalFactors{B, A}
    β :: B
    a :: A
end

Adapt.adapt_structure(to, factors::TridiagonalFactors) =
    TridiagonalFactors(adapt(to, factors.β), adapt(to, factors.a))

function Base.summary(solver::BatchedTridiagonalSolver)
    dirstr = prettysummary(solver.tridiagonal_direction)
    return "BatchedTridiagonalSolver in $dirstr"
end

function Base.show(io::IO, solver::BatchedTridiagonalSolver)
    print(io, summary(solver), '\n')
    print(io, "├── grid: ", prettysummary(solver.grid), '\n')
    print(io, "└── factorized: ", !isnothing(solver.factors))
end

# Some aliases...
const XTridiagonalSolver = BatchedTridiagonalSolver{A, B, C, T, G, P, <:XDirection} where {A, B, C, T, G, P}
const YTridiagonalSolver = BatchedTridiagonalSolver{A, B, C, T, G, P, <:YDirection} where {A, B, C, T, G, P}
const ZTridiagonalSolver = BatchedTridiagonalSolver{A, B, C, T, G, P, <:ZDirection} where {A, B, C, T, G, P}

Architectures.architecture(solver::BatchedTridiagonalSolver) = architecture(solver.grid)

"""
$(MethodSignatures(; defaults = true))

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

Stored factorization
====================

With `factorized = true`, the solver stores the factorization of its systems, which depends only
on the coefficients `a`, `b`, and `c` and not on the right-hand side. [`compute_factorization!`](@ref)
evaluates the coefficients and computes the pivots

```
    βⁱʲ¹ = bⁱʲ¹,
    βⁱʲᵏ = bⁱʲᵏ - aⁱʲᵏ⁻¹ tⁱʲᵏ,   tⁱʲᵏ = cⁱʲᵏ⁻¹ / βⁱʲᵏ⁻¹,   k = 2, ..., N,
```

and stores `β`, `t`, and `a`. `solve!` then only substitutes: it reads the stored factors, evaluates
neither the coefficients nor the recurrence, and reads only the right-hand side per level. This is
for systems solved many times with unchanged coefficients and different right-hand sides.

The caller calls `compute_factorization!` before the first `solve!` and again whenever the
coefficients change, including through `args`. A factorized `solve!` does not detect changed
coefficients and solves the system that was last factored. The factorization uses the same
floating-point operations as the unfactored sweep, so the solution is bitwise identical to that of
the default, `factorized = false`.

"""
function BatchedTridiagonalSolver(grid;
                                  lower_diagonal,
                                  diagonal,
                                  upper_diagonal,
                                  scratch = zeros(architecture(grid), eltype(grid), worksize(grid)...),
                                  parameters = nothing,
                                  tridiagonal_direction = ZDirection(),
                                  factorized = false)

    factors = factorized ? TridiagonalFactors(similar(scratch), similar(scratch)) : nothing

    return BatchedTridiagonalSolver(lower_diagonal, diagonal, upper_diagonal,
                                    scratch, grid, parameters, tridiagonal_direction, factors)
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

    launch_config = main_diagonal_launch_configuration(solver.tridiagonal_direction)

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
            solver.factors)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Compute and store the factorization of the systems of the factorized `solver` (see
[`BatchedTridiagonalSolver`](@ref)) from the current coefficients, evaluated with `args...`.
Call it before the first [`solve!`](@ref) and whenever the coefficients change.
"""
function compute_factorization!(solver::BatchedTridiagonalSolver, args...)
    isnothing(solver.factors) &&
        throw(ArgumentError("The solver was built with `factorized = false` and stores no factorization."))

    launch_config = main_diagonal_launch_configuration(solver.tridiagonal_direction)

    launch!(architecture(solver), solver.grid, launch_config,
            compute_batched_tridiagonal_factorization_kernel!,
            solver.factors,
            solver.a,
            solver.b,
            solver.c,
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

# The quantities of the elimination: computed from the coefficients (`factors = nothing`), or read
# from a stored factorization. Only the unfactored ratio is written to the scratch `t`.
@inline lower_diagonal(i, j, k, grid, ::Nothing, a, p, tridiagonal_direction, args) =
    get_coefficient(i, j, k, grid, a, p, tridiagonal_direction, args...)

@inline lower_diagonal(i, j, k, grid, factors::TridiagonalFactors, args...) = @inbounds factors.a[i, j, k]

@inline function elimination_ratio!(t, i, j, k, ::Nothing, i⁻, j⁻, k⁻, grid, c, p, tridiagonal_direction, args, β)
    cᵏ⁻¹ = get_coefficient(i⁻, j⁻, k⁻, grid, c, p, tridiagonal_direction, args...)
    tᵏ = convert(eltype(t), cᵏ⁻¹ / β)
    @inbounds t[i, j, k] = tᵏ
    return tᵏ
end

@inline elimination_ratio!(t, i, j, k, ::TridiagonalFactors, args...) = @inbounds t[i, j, k]

# Pivot `βᵏ = bᵏ - aᵏ⁻¹ tᵏ`
@inline pivot(i, j, k, grid, ::Nothing, b, p, tridiagonal_direction, args, aᵏ⁻¹, tᵏ) =
    get_coefficient(i, j, k, grid, b, p, tridiagonal_direction, args...) -
    off_diagonal_multiply(tridiagonal_direction, aᵏ⁻¹, tᵏ)

@inline pivot(i, j, k, grid, factors::TridiagonalFactors, args...) = @inbounds factors.β[i, j, k]

@kernel function compute_batched_tridiagonal_factorization_kernel!(factors, a, b, c, t, grid, p, args, tridiagonal_direction)
    m₁, m₂ = @index(Global, NTuple)
    N = size(grid, dimension(tridiagonal_direction))
    compute_batched_tridiagonal_factorization!(m₁, m₂, N, factors, a, b, c, t, grid, p, args, tridiagonal_direction)
end

# The forward recurrence of the sweep, without a right-hand side: stores `β`, `t`, and `a`
@inline function compute_batched_tridiagonal_factorization!(m₁, m₂, N, factors, a, b, c, t, grid, p, args,
                                                            tridiagonal_direction)
    Τ = eltype(t)

    @inbounds begin
        i, j, k = column_index(1, m₁, m₂, tridiagonal_direction)
        β = pivot(i, j, k, grid, nothing, b, p, tridiagonal_direction, args, zero(Τ), zero(Τ))
        factors.β[i, j, k] = β

        for n = 2:N
            i⁻, j⁻, k⁻ = column_index(n-1, m₁, m₂, tridiagonal_direction)
            i,  j,  k  = column_index(n,   m₁, m₂, tridiagonal_direction)

            aᵏ⁻¹ = lower_diagonal(i⁻, j⁻, k⁻, grid, nothing, a, p, tridiagonal_direction, args)
            tᵏ = elimination_ratio!(t, i, j, k, nothing, i⁻, j⁻, k⁻, grid, c, p, tridiagonal_direction, args, β)
            β = pivot(i, j, k, grid, nothing, b, p, tridiagonal_direction, args, aᵏ⁻¹, tᵏ)

            factors.a[i⁻, j⁻, k⁻] = aᵏ⁻¹
            factors.β[i, j, k] = β
        end
    end
end

@kernel function solve_batched_tridiagonal_system_kernel!(ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction, factors)
    m₁, m₂ = @index(Global, NTuple)
    N = size(grid, dimension(tridiagonal_direction))
    solve_batched_tridiagonal_system!(m₁, m₂, N, ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction, factors)
end

# Thomas sweep along column `(m₁, m₂)`. β, tᵏ and the last solved level are carried in local
# variables rather than re-read from `t` and `ϕ`, which the compiler cannot prove are unaliased.
@inline function solve_batched_tridiagonal_system!(m₁, m₂, N, ϕ, a, b, c, f, t, grid, p, args,
                                                   tridiagonal_direction, factors)
    Φ = eltype(ϕ)
    Τ = eltype(t)

    @inbounds begin
        i, j, k = column_index(1, m₁, m₂, tridiagonal_direction)
        β  = pivot(i, j, k, grid, factors, b, p, tridiagonal_direction, args, zero(Τ), zero(Τ))
        f₁ = get_coefficient(i, j, k, grid, f, p, tridiagonal_direction, args...)
        ϕ⁻ = convert(Φ, f₁ / β)
        ϕ[i, j, k] = ϕ⁻

        for n = 2:N
            i⁻, j⁻, k⁻ = column_index(n-1, m₁, m₂, tridiagonal_direction)
            i,  j,  k  = column_index(n,   m₁, m₂, tridiagonal_direction)

            aᵏ⁻¹ = lower_diagonal(i⁻, j⁻, k⁻, grid, factors, a, p, tridiagonal_direction, args)
            tᵏ = elimination_ratio!(t, i, j, k, factors, i⁻, j⁻, k⁻, grid, c, p, tridiagonal_direction, args, β)
            β = pivot(i, j, k, grid, factors, b, p, tridiagonal_direction, args, aᵏ⁻¹, tᵏ)

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
