using Oceananigans.Architectures: on_architecture
using Oceananigans.Grids: XDirection, YDirection, ZDirection

#####
##### Shared helpers for batched column solvers (tridiagonal and pentadiagonal)
#####

# A batched column solver sweeps along one `direction`; the other two indices identify the column
# and are supplied by the `:yz`, `:xz`, or `:xy` launch configuration. `column_indices` maps the
# sweep index `n` and the column indices `(p, q)` to the grid indices `(i, j, k)`.
@inline column_indices(::XDirection, n, p, q) = (n, p, q)
@inline column_indices(::YDirection, n, p, q) = (p, n, q)
@inline column_indices(::ZDirection, n, p, q) = (p, q, n)

launch_configuration(::XDirection) = :yz
launch_configuration(::YDirection) = :xz
launch_configuration(::ZDirection) = :xy

sweep_dimension(::XDirection) = 1
sweep_dimension(::YDirection) = 2
sweep_dimension(::ZDirection) = 3

# Size of an array holding one value per unknown of every system: `N` unknowns along `direction`.
function system_size(grid, direction, N)
    Nx, Ny, Nz = worksize(grid)
    dim = sweep_dimension(direction)
    return (dim == 1 ? N : Nx, dim == 2 ? N : Ny, dim == 3 ? N : Nz)
end

# `a * b` if `a != 0`, 0 otherwise (masks away NaNs in `b`)
@inline masked_multiply(x, y) = x * y * (x != 0)

# Only the `ZDirection` sweeps mask; see `solve!`.
@inline masked_multiply(x, y, ::ZDirection) = masked_multiply(x, y)
@inline masked_multiply(x, y, direction) = x * y

@inline float_eltype(ϕ::AbstractArray{T}) where T <: AbstractFloat = T
@inline float_eltype(ϕ::AbstractArray{<:Complex{T}}) where T <: AbstractFloat = T

#####
##### Batched tridiagonal solver
#####

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

function Base.summary(solver::BatchedTridiagonalSolver)
    dirstr = prettysummary(solver.tridiagonal_direction)
    return "BatchedTridiagonalSolver in $dirstr"
end

function Base.show(io::IO, solver::BatchedTridiagonalSolver)
    print(io, summary(solver), '\n')
    print(io, "├── grid: ", prettysummary(solver.grid), '\n')
    print(io, "└── factors: ", prettysummary(solver.factors))
end

# Some aliases...
const XTridiagonalSolver = BatchedTridiagonalSolver{A, B, C, T, G, P, <:XDirection} where {A, B, C, T, G, P}
const YTridiagonalSolver = BatchedTridiagonalSolver{A, B, C, T, G, P, <:YDirection} where {A, B, C, T, G, P}
const ZTridiagonalSolver = BatchedTridiagonalSolver{A, B, C, T, G, P, <:ZDirection} where {A, B, C, T, G, P}

Architectures.architecture(solver::BatchedTridiagonalSolver) = architecture(solver.grid)

"""
    struct TridiagonalFactors{B, L, T}

Storage for the LU factors of the systems of a [`BatchedTridiagonalSolver`](@ref), filled by
[`factorize!`](@ref) and consumed by [`solve_factorized!`](@ref). For every row `n` of every
system, `β[n]` is the pivot (the diagonal of `U`), `lower[n]` is the lower-diagonal coefficient
of row `n` (that is, `aⁿ⁻¹` in the convention of [`BatchedTridiagonalSolver`](@ref)), and `t[n]`
is the super-diagonal of the unit upper factor, `t[n] = cⁿ⁻¹ / β[n-1]`.
"""
struct TridiagonalFactors{B, L, T}
    β :: B
    lower :: L
    t :: T
end

TridiagonalFactors(arch::AbstractArchitecture, FT, sz::Tuple) =
    TridiagonalFactors(zeros(arch, FT, sz...), zeros(arch, FT, sz...), zeros(arch, FT, sz...))

Base.summary(factors::TridiagonalFactors) = "TridiagonalFactors of size $(size(factors.β))"

"""
    BatchedTridiagonalSolver(grid;
                             lower_diagonal,
                             diagonal,
                             upper_diagonal,
                             scratch = zeros(architecture(grid), eltype(grid), grid.Nx, grid.Ny, grid.Nz),
                             tridiagonal_direction = ZDirection(),
                             parameters = nothing,
                             store_factors = false)

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

With `store_factors = true`, the solver also allocates a [`TridiagonalFactors`](@ref) (three arrays
of the same size as `scratch`), so that the coefficients can be evaluated once with
[`factorize!`](@ref) and the systems then solved repeatedly, for different right hand sides,
with [`solve_factorized!`](@ref).
"""
function BatchedTridiagonalSolver(grid;
                                  lower_diagonal,
                                  diagonal,
                                  upper_diagonal,
                                  scratch = zeros(architecture(grid), eltype(grid), worksize(grid)...),
                                  parameters = nothing,
                                  tridiagonal_direction = ZDirection(),
                                  store_factors = false)

    factors = store_factors ? TridiagonalFactors(architecture(grid), eltype(grid), size(scratch)) : nothing

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

    launch!(architecture(solver), solver.grid, launch_configuration(solver.tridiagonal_direction),
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

@kernel function solve_batched_tridiagonal_system_kernel!(ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction::XDirection)
    Nx = size(grid, 1)
    j, k = @index(Global, NTuple)
    solve_batched_tridiagonal_system_x!(j, k, Nx, ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction)
end

@inline function solve_batched_tridiagonal_system_x!(j, k, Nx, ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction)
    @inbounds begin
        β  = get_coefficient(1, j, k, grid, b, p, tridiagonal_direction, args...)
        f₁ = get_coefficient(1, j, k, grid, f, p, tridiagonal_direction, args...)
        ϕ[1, j, k] = f₁ / β

        for i = 2:Nx
            cᵏ⁻¹ = get_coefficient(i-1, j, k, grid, c, p, tridiagonal_direction, args...)
            bᵏ   = get_coefficient(i,   j, k, grid, b, p, tridiagonal_direction, args...)
            aᵏ⁻¹ = get_coefficient(i-1, j, k, grid, a, p, tridiagonal_direction, args...)

            t[i, j, k] = cᵏ⁻¹ / β
            β = bᵏ - aᵏ⁻¹ * t[i, j, k]

            fᵏ = get_coefficient(i, j, k, grid, f, p, tridiagonal_direction, args...)

            # If the problem is not diagonally-dominant such that `β ≈ 0`,
            # the algorithm is unstable and we store the right-hand side instead.
            definitely_diagonally_dominant = abs(β) > 10 * eps(float_eltype(ϕ))
            ϕ★ = (fᵏ - aᵏ⁻¹ * ϕ[i-1, j, k]) / β
            ϕ[i, j, k] = ifelse(definitely_diagonally_dominant, ϕ★, fᵏ)
        end

        for i = Nx-1:-1:1
            ϕ[i, j, k] -= t[i+1, j, k] * ϕ[i+1, j, k]
        end
    end
end

@kernel function solve_batched_tridiagonal_system_kernel!(ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction::YDirection)
    Ny = size(grid, 2)
    i, k = @index(Global, NTuple)
    solve_batched_tridiagonal_system_y!(i, k, Ny, ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction)
end

@inline function solve_batched_tridiagonal_system_y!(i, k, Ny, ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction)
    @inbounds begin
        β  = get_coefficient(i, 1, k, grid, b, p, tridiagonal_direction, args...)
        f₁ = get_coefficient(i, 1, k, grid, f, p, tridiagonal_direction, args...)
        ϕ[i, 1, k] = f₁ / β

        for j = 2:Ny
            cᵏ⁻¹ = get_coefficient(i, j-1, k, grid, c, p, tridiagonal_direction, args...)
            bᵏ   = get_coefficient(i, j,   k, grid, b, p, tridiagonal_direction, args...)
            aᵏ⁻¹ = get_coefficient(i, j-1, k, grid, a, p, tridiagonal_direction, args...)

            t[i, j, k] = cᵏ⁻¹ / β
            β = bᵏ - aᵏ⁻¹ * t[i, j, k]

            fᵏ = get_coefficient(i, j, k, grid, f, p, tridiagonal_direction, args...)

            # If the problem is not diagonally-dominant such that `β ≈ 0`,
            # the algorithm is unstable and we store the right-hand side instead.
            definitely_diagonally_dominant = abs(β) > 10 * eps(float_eltype(ϕ))
            ϕ★ = (fᵏ - aᵏ⁻¹ * ϕ[i, j-1, k]) / β
            ϕ[i, j, k] = ifelse(definitely_diagonally_dominant, ϕ★, fᵏ)
        end

        for j = Ny-1:-1:1
            ϕ[i, j, k] -= t[i, j+1, k] * ϕ[i, j+1, k]
        end
    end
end

@kernel function solve_batched_tridiagonal_system_kernel!(ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction::ZDirection)
    Nz = size(grid, 3)
    i, j = @index(Global, NTuple)
    solve_batched_tridiagonal_system_z!(i, j, Nz, ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction)
end

@inline function solve_batched_tridiagonal_system_z!(i, j, Nz, ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction)
    @inbounds begin
        β  = get_coefficient(i, j, 1, grid, b, p, tridiagonal_direction, args...)
        f₁ = get_coefficient(i, j, 1, grid, f, p, tridiagonal_direction, args...)
        ϕ[i, j, 1] = f₁ / β

        for k = 2:Nz
            cᵏ⁻¹ = get_coefficient(i, j, k-1, grid, c, p, tridiagonal_direction, args...)
            bᵏ   = get_coefficient(i, j, k,   grid, b, p, tridiagonal_direction, args...)
            aᵏ⁻¹ = get_coefficient(i, j, k-1, grid, a, p, tridiagonal_direction, args...)

            t[i, j, k] = cᵏ⁻¹ / β
            β = bᵏ - masked_multiply(aᵏ⁻¹, t[i, j, k])
            fᵏ = get_coefficient(i, j, k, grid, f, p, tridiagonal_direction, args...)

            # If the problem is not diagonally-dominant such that `β ≈ 0`,
            # the algorithm is unstable and we store the right-hand side instead.
            definitely_diagonally_dominant = abs(β) > 10 * eps(float_eltype(ϕ))
            ϕ★ = (fᵏ - masked_multiply(aᵏ⁻¹, ϕ[i, j, k-1])) / β
            ϕ[i, j, k] = ifelse(definitely_diagonally_dominant, ϕ★, fᵏ)
        end

        for k = Nz-1:-1:1
            ϕ[i, j, k] -= masked_multiply(t[i, j, k+1], ϕ[i, j, k+1])
        end
    end
end

#####
##### Factor once, substitute many
#####

"""
$(TYPEDSIGNATURES)

Evaluate the coefficients of the batched tridiagonal systems described by `solver` once, passing
`args...` to `get_coefficient`, and store their LU factors in `solver.factors` (see
[`TridiagonalFactors`](@ref)). The systems can then be solved for any number of right hand sides
with [`solve_factorized!`](@ref), which only performs the forward and backward substitutions.
The solver must have been constructed with `store_factors = true`.

```jldoctest
julia> using Oceananigans, Oceananigans.Solvers

julia> grid = RectilinearGrid(size=(1, 1, 4), extent=(1, 1, 1));

julia> solver = BatchedTridiagonalSolver(grid; lower_diagonal = [1, 1, 1], diagonal = [4, 4, 4, 4],
                                         upper_diagonal = [1, 1, 1], store_factors = true)
BatchedTridiagonalSolver in ZDirection()
├── grid: 1×1×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
└── factors: TridiagonalFactors of size (1, 1, 4)

julia> factorize!(solver);

julia> ϕ = zeros(1, 1, 4); rhs = ones(1, 1, 4);

julia> solve_factorized!(ϕ, solver, rhs);

julia> ϕ[:]
4-element Vector{Float64}:
 0.21052631578947367
 0.15789473684210525
 0.15789473684210528
 0.21052631578947367
```
"""
function factorize!(solver::BatchedTridiagonalSolver, args...)
    factors = solver.factors
    isnothing(factors) && throw(ArgumentError("The solver has no storage for factors; construct it with `store_factors = true`."))
    direction = solver.tridiagonal_direction
    N = size(solver.grid, sweep_dimension(direction))

    launch!(architecture(solver), solver.grid, launch_configuration(direction),
            factorize_batched_tridiagonal_system_kernel!,
            factors.β, factors.lower, factors.t, N,
            solver.a, solver.b, solver.c,
            solver.grid, solver.parameters, Tuple(args), direction)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Solve the batched tridiagonal systems whose factors were stored in `solver.factors` by
[`factorize!`](@ref), for the right hand side `rhs` (passing `args...` to `get_coefficient`
for `rhs` only). The result is stored in `ϕ`.

The arithmetic is operation for operation the same as in [`solve!`](@ref), including the
masking of the `ZDirection` sweep and the fallback to the right hand side where a pivot
vanishes, so that `solve_factorized!` reproduces `solve!` bitwise for the same coefficients.
"""
function solve_factorized!(ϕ, solver::BatchedTridiagonalSolver, rhs, args...)
    factors = solver.factors
    isnothing(factors) && throw(ArgumentError("The solver has no factors; construct it with `store_factors = true` and call `factorize!`."))
    direction = solver.tridiagonal_direction
    N = size(solver.grid, sweep_dimension(direction))

    launch!(architecture(solver), solver.grid, launch_configuration(direction),
            substitute_batched_tridiagonal_system_kernel!,
            ϕ, factors.β, factors.lower, factors.t, N, rhs,
            solver.grid, solver.parameters, Tuple(args), direction)

    return nothing
end

@kernel function factorize_batched_tridiagonal_system_kernel!(β, lower, t, N, a, b, c, grid, parameters, args, direction)
    p, q = @index(Global, NTuple)
    factorize_tridiagonal_column!(p, q, β, lower, t, N, a, b, c, grid, parameters, args, direction)
end

# Mirrors the forward sweep of `solve_batched_tridiagonal_system_z!` (and its x and y counterparts),
# storing the pivot `β`, the lower coefficient `aⁿ⁻¹` of each row and `t = cⁿ⁻¹ / βⁿ⁻¹`.
@inline function factorize_tridiagonal_column!(p, q, β, lower, t, N, a, b, c, grid, parameters, args, direction)
    @inbounds begin
        i, j, k = column_indices(direction, 1, p, q)
        βⁿ = get_coefficient(i, j, k, grid, b, parameters, direction, args...)
        β[i, j, k] = βⁿ
        lower[i, j, k] = 0
        t[i, j, k] = 0

        for n = 2:N
            i⁻, j⁻, k⁻ = column_indices(direction, n-1, p, q)
            i, j, k = column_indices(direction, n, p, q)

            cⁿ⁻¹ = get_coefficient(i⁻, j⁻, k⁻, grid, c, parameters, direction, args...)
            bⁿ   = get_coefficient(i,  j,  k,  grid, b, parameters, direction, args...)
            aⁿ⁻¹ = get_coefficient(i⁻, j⁻, k⁻, grid, a, parameters, direction, args...)

            tⁿ = cⁿ⁻¹ / βⁿ
            βⁿ = bⁿ - masked_multiply(aⁿ⁻¹, tⁿ, direction)

            t[i, j, k] = tⁿ
            β[i, j, k] = βⁿ
            lower[i, j, k] = aⁿ⁻¹
        end
    end
end

@kernel function substitute_batched_tridiagonal_system_kernel!(ϕ, β, lower, t, N, f, grid, parameters, args, direction)
    p, q = @index(Global, NTuple)
    substitute_tridiagonal_column!(p, q, ϕ, β, lower, t, N, f, grid, parameters, args, direction)
end

@inline function substitute_tridiagonal_column!(p, q, ϕ, β, lower, t, N, f, grid, parameters, args, direction)
    @inbounds begin
        i, j, k = column_indices(direction, 1, p, q)
        f₁ = get_coefficient(i, j, k, grid, f, parameters, direction, args...)
        ϕ[i, j, k] = f₁ / β[i, j, k]

        for n = 2:N
            i⁻, j⁻, k⁻ = column_indices(direction, n-1, p, q)
            i, j, k = column_indices(direction, n, p, q)

            βⁿ   = β[i, j, k]
            aⁿ⁻¹ = lower[i, j, k]
            fⁿ   = get_coefficient(i, j, k, grid, f, parameters, direction, args...)

            # If the problem is not diagonally-dominant such that `β ≈ 0`,
            # the algorithm is unstable and we store the right-hand side instead.
            definitely_diagonally_dominant = abs(βⁿ) > 10 * eps(float_eltype(ϕ))
            ϕ★ = (fⁿ - masked_multiply(aⁿ⁻¹, ϕ[i⁻, j⁻, k⁻], direction)) / βⁿ
            ϕ[i, j, k] = ifelse(definitely_diagonally_dominant, ϕ★, fⁿ)
        end

        for n = N-1:-1:1
            i⁺, j⁺, k⁺ = column_indices(direction, n+1, p, q)
            i, j, k = column_indices(direction, n, p, q)
            ϕ[i, j, k] -= masked_multiply(t[i⁺, j⁺, k⁺], ϕ[i⁺, j⁺, k⁺], direction)
        end
    end
end
