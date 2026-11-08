#####
##### Batched pentadiagonal solver: banded LU without pivoting, one thread per system
#####

"""
    struct BatchedPentadiagonalSolver{E, A, B, C, D, S, G, P, Dir, F}

A batched solver for large numbers of pentadiagonal systems.
"""
struct BatchedPentadiagonalSolver{E, A, B, C, D, S, G, P, Dir, F}
    e :: E
    a :: A
    b :: B
    c :: C
    d :: D
    scratch :: S
    N :: Int
    grid :: G
    parameters :: P
    direction :: Dir
    factors :: F
end

function Base.summary(solver::BatchedPentadiagonalSolver)
    dirstr = prettysummary(solver.direction)
    return "BatchedPentadiagonalSolver in $dirstr with N = $(solver.N)"
end

function Base.show(io::IO, solver::BatchedPentadiagonalSolver)
    print(io, summary(solver), '\n')
    print(io, "├── grid: ", prettysummary(solver.grid), '\n')
    print(io, "└── factors: ", prettysummary(solver.factors))
end

Architectures.architecture(solver::BatchedPentadiagonalSolver) = architecture(solver.grid)

"""
    struct PentadiagonalFactors{L, U, C, D}

Storage for the LU factors of the systems of a [`BatchedPentadiagonalSolver`](@ref), filled by
[`factorize!`](@ref) and consumed by [`solve_factorized!`](@ref). `A = L U` with a unit lower
factor `L` with bands `l₁[n] = L[n, n-1]` and `l₂[n] = L[n, n-2]`, and an upper factor `U` with
diagonal `u[n]` (stored as its reciprocal `u⁻¹`), first super-diagonal `c̃[n] = U[n, n+1]`, and
second super-diagonal `d[n] = U[n, n+2]` (equal to the second upper diagonal of `A`).
"""
struct PentadiagonalFactors{L, U, C, D}
    l₁ :: L
    l₂ :: L
    u⁻¹ :: U
    c̃ :: C
    d :: D
end

function PentadiagonalFactors(arch::AbstractArchitecture, FT, sz::Tuple)
    l₁  = zeros(arch, FT, sz...)
    l₂  = zeros(arch, FT, sz...)
    u⁻¹ = zeros(arch, FT, sz...)
    c̃   = zeros(arch, FT, sz...)
    d   = zeros(arch, FT, sz...)
    return PentadiagonalFactors(l₁, l₂, u⁻¹, c̃, d)
end

Base.summary(factors::PentadiagonalFactors) = "PentadiagonalFactors of size $(size(factors.u⁻¹))"

"""
    BatchedPentadiagonalSolver(grid;
                               second_lower_diagonal,
                               lower_diagonal,
                               diagonal,
                               upper_diagonal,
                               second_upper_diagonal,
                               direction = ZDirection(),
                               N = size(grid, sweep_dimension(direction)),
                               scratch = (zeros(architecture(grid), eltype(grid), system_size(grid, direction, N)...),
                                          zeros(architecture(grid), eltype(grid), system_size(grid, direction, N)...)),
                               parameters = nothing,
                               store_factors = false)

Construct a solver for batched pentadiagonal systems on `grid` of the form

```
    eⁱʲⁿ ϕⁱʲⁿ⁻² + aⁱʲⁿ ϕⁱʲⁿ⁻¹ + bⁱʲⁿ ϕⁱʲⁿ + cⁱʲⁿ ϕⁱʲⁿ⁺¹ + dⁱʲⁿ ϕⁱʲⁿ⁺² = fⁱʲⁿ,  n = 1, ..., N
```
or in matrix form
```
    ⎡ bⁱʲ¹   cⁱʲ¹   dⁱʲ¹     0      ⋯        0    ⎤ ⎡ ϕⁱʲ¹ ⎤   ⎡ fⁱʲ¹ ⎤
    ⎢ aⁱʲ²   bⁱʲ²   cⁱʲ²   dⁱʲ²     0        ⋮    ⎥ ⎢ ϕⁱʲ² ⎥   ⎢ fⁱʲ² ⎥
    ⎢ eⁱʲ³   aⁱʲ³   bⁱʲ³   cⁱʲ³   dⁱʲ³            ⎥ ⎢ ϕⁱʲ³ ⎥   ⎢ fⁱʲ³ ⎥
    ⎢  0      ⋱      ⋱      ⋱      ⋱        0    ⎥ ⎢   .  ⎥ = ⎢   .  ⎥
    ⎢  ⋮            eⁱʲᴺ⁻¹ aⁱʲᴺ⁻¹  bⁱʲᴺ⁻¹  cⁱʲᴺ⁻¹ ⎥ ⎢      ⎥   ⎢   .  ⎥
    ⎣  0      ⋯       0     eⁱʲᴺ   aⁱʲᴺ    bⁱʲᴺ  ⎦ ⎣ ϕⁱʲᴺ ⎦   ⎣ fⁱʲᴺ ⎦
```

where `e` is the `second_lower_diagonal`, `a` the `lower_diagonal`, `b` the `diagonal`,
`c` the `upper_diagonal`, and `d` the `second_upper_diagonal`.

!!! note "Index convention"
    All five bands are indexed *by row*: `eⁿ`, `aⁿ`, `bⁿ`, `cⁿ`, and `dⁿ` are the coefficients
    of the `n`-th equation, so that `eⁿ = A[n, n-2]`, `aⁿ = A[n, n-1]`, `cⁿ = A[n, n+1]`, and
    `dⁿ = A[n, n+2]`. This differs from [`BatchedTridiagonalSolver`](@ref), whose lower diagonal
    `aⁿ` multiplies `ϕⁿ` in row `n+1` (the convention of Press et al. 1992), and from
    `LinearAlgebra.Tridiagonal`, whose off-diagonals have `N-1` entries. Here every band has
    `N` entries along the solve direction, and the entries that fall outside the matrix
    (`e¹`, `e²`, `a¹`, `cᴺ`, `dᴺ⁻¹`, `dᴺ`) are ignored: they may hold any value, but their
    storage must exist.

The systems are solved along `direction` (one of `XDirection()`, `YDirection()`, `ZDirection()`)
with `N` unknowns each. `N` defaults to the grid size in that direction, but may differ from it:
for example, two unknowns per cell interleaved as `(ϕ₁¹, ϕ₂¹, ϕ₁², ϕ₂², …)` give a `2 Nz` system
per column, which is pentadiagonal when the unknowns are coupled across adjacent cells.
`ϕ`, `f` and array coefficients then have `N` entries along `direction` (for example, size
`(Nx, Ny, N)` in the `ZDirection`).

`e`, `a`, `b`, `c`, `d`, and `f` can be specified in three ways:

1. A 1D array means, e.g., that `aⁱʲⁿ = a[n]`.

2. A 3D array means, e.g., that `aⁱʲⁿ = a[i, j, n]`.

3. Other coefficient types can be implemented by extending `get_coefficient`; the coefficient
   of row `n` of the system in column `(i, j)` is `get_coefficient(i, j, n, grid, a, parameters, direction, args...)`,
   where `args...` are the trailing arguments of `solve!` or `factorize!`.

The systems are factored by banded LU *without pivoting*, which is stable for diagonally dominant
systems (and, more generally, for systems whose LU factorization is stable without pivoting).
Unlike [`BatchedTridiagonalSolver`](@ref), no masking of vanishing off-diagonals is performed.
`scratch` holds two arrays of the size of `ϕ` used by [`solve!`](@ref) to store the factors of
the current system during the sweep. With `store_factors = true`, the solver also allocates a
[`PentadiagonalFactors`](@ref) (five arrays of the size of `ϕ`), so that the coefficients can be
evaluated once with [`factorize!`](@ref) and the systems then solved repeatedly, for different
right hand sides, with [`solve_factorized!`](@ref).

```jldoctest
julia> using Oceananigans, Oceananigans.Solvers

julia> grid = RectilinearGrid(size=(1, 1, 3), extent=(1, 1, 1));

julia> solver = BatchedPentadiagonalSolver(grid; N = 6,
                                           second_lower_diagonal = fill(0.1, 6),
                                           lower_diagonal = fill(1.0, 6),
                                           diagonal = fill(4.0, 6),
                                           upper_diagonal = fill(1.0, 6),
                                           second_upper_diagonal = fill(0.1, 6))
BatchedPentadiagonalSolver in ZDirection() with N = 6
├── grid: 1×1×3 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
└── factors: nothing
```
"""
function BatchedPentadiagonalSolver(grid;
                                    second_lower_diagonal,
                                    lower_diagonal,
                                    diagonal,
                                    upper_diagonal,
                                    second_upper_diagonal,
                                    direction = ZDirection(),
                                    N = size(grid, sweep_dimension(direction)),
                                    scratch = (zeros(architecture(grid), eltype(grid), system_size(grid, direction, N)...),
                                               zeros(architecture(grid), eltype(grid), system_size(grid, direction, N)...)),
                                    parameters = nothing,
                                    store_factors = false)

    N ≥ 3 || throw(ArgumentError("A pentadiagonal system needs at least N = 3 unknowns (got N = $N)."))
    factors = store_factors ? PentadiagonalFactors(architecture(grid), eltype(grid), size(first(scratch))) : nothing

    return BatchedPentadiagonalSolver(second_lower_diagonal, lower_diagonal, diagonal, upper_diagonal, second_upper_diagonal,
                                      scratch, N, grid, parameters, direction, factors)
end

"""
$(TYPEDSIGNATURES)

Solve the batched pentadiagonal systems with right hand side `rhs` and coefficients described by
the `BatchedPentadiagonalSolver` `solver`, evaluating the coefficients on the fly and passing
`args...` to `get_coefficient`. The result is stored in `ϕ`, which has `solver.N` entries along
the solve direction.

Each system is factored as `A = L U` by banded Gaussian elimination without pivoting, where for
row `n` (with `l₂ = e / u`, `l₁ = (a - l₂ c̃) / u`, `u = b - l₂ d - l₁ c̃`, `c̃ = c - l₁ d`, taking
the quantities of rows `n-1` and `n-2` where appropriate), followed by the forward substitution
`yⁿ = fⁿ - l₁ⁿ yⁿ⁻¹ - l₂ⁿ yⁿ⁻²` and the backward substitution
`ϕⁿ = (yⁿ - c̃ⁿ ϕⁿ⁺¹ - dⁿ ϕⁿ⁺²) / uⁿ`. For unknowns interleaved in pairs, this is exactly block
tridiagonal elimination with 2×2 blocks.
"""
function solve!(ϕ, solver::BatchedPentadiagonalSolver, rhs, args...)
    u⁻¹, c̃ = solver.scratch

    launch!(architecture(solver), solver.grid, launch_configuration(solver.direction),
            solve_batched_pentadiagonal_system_kernel!,
            ϕ, solver.e, solver.a, solver.b, solver.c, solver.d, rhs, u⁻¹, c̃, solver.N,
            solver.grid, solver.parameters, Tuple(args), solver.direction)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Evaluate the coefficients of the batched pentadiagonal systems described by `solver` once,
passing `args...` to `get_coefficient`, and store their LU factors in `solver.factors` (see
[`PentadiagonalFactors`](@ref)). The systems can then be solved for any number of right hand
sides with [`solve_factorized!`](@ref). The solver must have been constructed with
`store_factors = true`.

```jldoctest
julia> using Oceananigans, Oceananigans.Solvers

julia> grid = RectilinearGrid(size=(1, 1, 5), extent=(1, 1, 1));

julia> solver = BatchedPentadiagonalSolver(grid; second_lower_diagonal = fill(0.1, 5),
                                           lower_diagonal = fill(1.0, 5), diagonal = fill(4.0, 5),
                                           upper_diagonal = fill(1.0, 5), second_upper_diagonal = fill(0.1, 5),
                                           store_factors = true);

julia> factorize!(solver);

julia> ϕ = zeros(1, 1, 5); solve_factorized!(ϕ, solver, ones(1, 1, 5));

julia> ϕ[:]
5-element Vector{Float64}:
 0.20753737156422716
 0.15356652694832895
 0.16283986794762415
 0.15356652694832892
 0.2075373715642272
```
"""
function factorize!(solver::BatchedPentadiagonalSolver, args...)
    factors = solver.factors
    isnothing(factors) && throw(ArgumentError("The solver has no storage for factors; construct it with `store_factors = true`."))

    launch!(architecture(solver), solver.grid, launch_configuration(solver.direction),
            factorize_batched_pentadiagonal_system_kernel!,
            factors.l₁, factors.l₂, factors.u⁻¹, factors.c̃, factors.d, solver.N,
            solver.e, solver.a, solver.b, solver.c, solver.d,
            solver.grid, solver.parameters, Tuple(args), solver.direction)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Solve the batched pentadiagonal systems whose factors were stored in `solver.factors` by
[`factorize!`](@ref), for the right hand side `rhs` (passing `args...` to `get_coefficient`
for `rhs` only). The result is stored in `ϕ`. The substitutions perform the same arithmetic
as [`solve!`](@ref), so the two agree bitwise for the same coefficients.
"""
function solve_factorized!(ϕ, solver::BatchedPentadiagonalSolver, rhs, args...)
    factors = solver.factors
    isnothing(factors) && throw(ArgumentError("The solver has no factors; construct it with `store_factors = true` and call `factorize!`."))

    launch!(architecture(solver), solver.grid, launch_configuration(solver.direction),
            substitute_batched_pentadiagonal_system_kernel!,
            ϕ, factors.l₁, factors.l₂, factors.u⁻¹, factors.c̃, factors.d, solver.N, rhs,
            solver.grid, solver.parameters, Tuple(args), solver.direction)

    return nothing
end

#####
##### Kernels
#####

# Coefficients of row `n` of the system in column `(p, q)`; the bands that fall outside the matrix are zero.
@inline function pentadiagonal_row(n, p, q, N, e, a, b, c, d, grid, parameters, args, direction)
    i, j, k = column_indices(direction, n, p, q)
    FT = eltype(grid)
    eⁿ = ifelse(n > 2,     get_coefficient(i, j, k, grid, e, parameters, direction, args...), zero(FT))
    aⁿ = ifelse(n > 1,     get_coefficient(i, j, k, grid, a, parameters, direction, args...), zero(FT))
    bⁿ =                   get_coefficient(i, j, k, grid, b, parameters, direction, args...)
    cⁿ = ifelse(n < N,     get_coefficient(i, j, k, grid, c, parameters, direction, args...), zero(FT))
    dⁿ = ifelse(n < N - 1, get_coefficient(i, j, k, grid, d, parameters, direction, args...), zero(FT))
    return eⁿ, aⁿ, bⁿ, cⁿ, dⁿ
end

# One step of the banded LU for row n, given the raw bands of row n and (u⁻¹, c̃, d) of rows n-1 and n-2.
# For n = 1, 2 the missing history is zero, so the formulas collapse to the right ones.
@inline function pentadiagonal_lu_step(eⁿ, aⁿ, bⁿ, cⁿ, u₁⁻¹, c̃₁, d₁, u₂⁻¹, c̃₂, d₂)
    l₂ = eⁿ * u₂⁻¹
    l₁ = (aⁿ - l₂ * c̃₂) * u₁⁻¹
    uⁿ = bⁿ - l₂ * d₂ - l₁ * c̃₁
    c̃ⁿ = cⁿ - l₁ * d₁
    return l₁, l₂, uⁿ, c̃ⁿ
end

@kernel function solve_batched_pentadiagonal_system_kernel!(ϕ, e, a, b, c, d, f, u⁻¹, c̃, N, grid, parameters, args, direction)
    p, q = @index(Global, NTuple)
    solve_pentadiagonal_column!(p, q, ϕ, e, a, b, c, d, f, u⁻¹, c̃, N, grid, parameters, args, direction)
end

@inline function solve_pentadiagonal_column!(p, q, ϕ, e, a, b, c, d, f, u⁻¹, c̃, N, grid, parameters, args, direction)
    FT = eltype(grid)
    T = eltype(ϕ)
    @inbounds begin
        # (u⁻¹, c̃, d, y) of rows n-1 (subscript 1) and n-2 (subscript 2)
        u₁⁻¹ = zero(FT); c̃₁ = zero(FT); d₁ = zero(FT); y₁ = zero(T)
        u₂⁻¹ = zero(FT); c̃₂ = zero(FT); d₂ = zero(FT); y₂ = zero(T)

        for n = 1:N
            i, j, k = column_indices(direction, n, p, q)
            eⁿ, aⁿ, bⁿ, cⁿ, dⁿ = pentadiagonal_row(n, p, q, N, e, a, b, c, d, grid, parameters, args, direction)
            fⁿ = get_coefficient(i, j, k, grid, f, parameters, direction, args...)

            l₁, l₂, uⁿ, c̃ⁿ = pentadiagonal_lu_step(eⁿ, aⁿ, bⁿ, cⁿ, u₁⁻¹, c̃₁, d₁, u₂⁻¹, c̃₂, d₂)
            yⁿ = fⁿ - l₁ * y₁ - l₂ * y₂
            uⁿ⁻¹ = 1 / uⁿ

            ϕ[i, j, k] = yⁿ
            u⁻¹[i, j, k] = uⁿ⁻¹
            c̃[i, j, k] = c̃ⁿ

            u₂⁻¹ = u₁⁻¹; c̃₂ = c̃₁; d₂ = d₁; y₂ = y₁
            u₁⁻¹ = uⁿ⁻¹; c̃₁ = c̃ⁿ; d₁ = dⁿ; y₁ = yⁿ
        end

        # ϕ of rows n+1 (subscript 1) and n+2 (subscript 2)
        ϕ₁ = zero(T); ϕ₂ = zero(T)

        for n = N:-1:1
            i, j, k = column_indices(direction, n, p, q)
            dⁿ = ifelse(n < N - 1, get_coefficient(i, j, k, grid, d, parameters, direction, args...), zero(FT))
            ϕⁿ = (ϕ[i, j, k] - c̃[i, j, k] * ϕ₁ - dⁿ * ϕ₂) * u⁻¹[i, j, k]
            ϕ[i, j, k] = ϕⁿ
            ϕ₂ = ϕ₁; ϕ₁ = ϕⁿ
        end
    end
end

@kernel function factorize_batched_pentadiagonal_system_kernel!(l₁, l₂, u⁻¹, c̃, d̃, N, e, a, b, c, d, grid, parameters, args, direction)
    p, q = @index(Global, NTuple)
    factorize_pentadiagonal_column!(p, q, l₁, l₂, u⁻¹, c̃, d̃, N, e, a, b, c, d, grid, parameters, args, direction)
end

@inline function factorize_pentadiagonal_column!(p, q, L₁, L₂, U⁻¹, C̃, D, N, e, a, b, c, d, grid, parameters, args, direction)
    FT = eltype(grid)
    @inbounds begin
        u₁⁻¹ = zero(FT); c̃₁ = zero(FT); d₁ = zero(FT)
        u₂⁻¹ = zero(FT); c̃₂ = zero(FT); d₂ = zero(FT)

        for n = 1:N
            i, j, k = column_indices(direction, n, p, q)
            eⁿ, aⁿ, bⁿ, cⁿ, dⁿ = pentadiagonal_row(n, p, q, N, e, a, b, c, d, grid, parameters, args, direction)

            l₁, l₂, uⁿ, c̃ⁿ = pentadiagonal_lu_step(eⁿ, aⁿ, bⁿ, cⁿ, u₁⁻¹, c̃₁, d₁, u₂⁻¹, c̃₂, d₂)
            uⁿ⁻¹ = 1 / uⁿ

            L₁[i, j, k] = l₁
            L₂[i, j, k] = l₂
            U⁻¹[i, j, k] = uⁿ⁻¹
            C̃[i, j, k] = c̃ⁿ
            D[i, j, k] = dⁿ

            u₂⁻¹ = u₁⁻¹; c̃₂ = c̃₁; d₂ = d₁
            u₁⁻¹ = uⁿ⁻¹; c̃₁ = c̃ⁿ; d₁ = dⁿ
        end
    end
end

@kernel function substitute_batched_pentadiagonal_system_kernel!(ϕ, l₁, l₂, u⁻¹, c̃, d, N, f, grid, parameters, args, direction)
    p, q = @index(Global, NTuple)
    substitute_pentadiagonal_column!(p, q, ϕ, l₁, l₂, u⁻¹, c̃, d, N, f, grid, parameters, args, direction)
end

@inline function substitute_pentadiagonal_column!(p, q, ϕ, L₁, L₂, U⁻¹, C̃, D, N, f, grid, parameters, args, direction)
    T = eltype(ϕ)
    @inbounds begin
        y₁ = zero(T); y₂ = zero(T)

        for n = 1:N
            i, j, k = column_indices(direction, n, p, q)
            fⁿ = get_coefficient(i, j, k, grid, f, parameters, direction, args...)
            yⁿ = fⁿ - L₁[i, j, k] * y₁ - L₂[i, j, k] * y₂
            ϕ[i, j, k] = yⁿ
            y₂ = y₁; y₁ = yⁿ
        end

        ϕ₁ = zero(T); ϕ₂ = zero(T)

        for n = N:-1:1
            i, j, k = column_indices(direction, n, p, q)
            ϕⁿ = (ϕ[i, j, k] - C̃[i, j, k] * ϕ₁ - D[i, j, k] * ϕ₂) * U⁻¹[i, j, k]
            ϕ[i, j, k] = ϕⁿ
            ϕ₂ = ϕ₁; ϕ₁ = ϕⁿ
        end
    end
end
