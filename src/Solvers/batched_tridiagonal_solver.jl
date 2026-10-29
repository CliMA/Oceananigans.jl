using Oceananigans.Architectures: on_architecture
using Oceananigans.Grids: XDirection, YDirection, ZDirection

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
"""
function BatchedTridiagonalSolver(grid;
                                  lower_diagonal,
                                  diagonal,
                                  upper_diagonal,
                                  scratch = zeros(architecture(grid), eltype(grid), worksize(grid)...),
                                  parameters = nothing,
                                  tridiagonal_direction = ZDirection())

    return BatchedTridiagonalSolver(lower_diagonal, diagonal, upper_diagonal,
                                    scratch, grid, parameters, tridiagonal_direction)
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
            # the algorithm is unstable and we elide the forward pass update of ϕ.
            definitely_diagonally_dominant = abs(β) > 10 * eps(float_eltype(ϕ))
            ϕ★ = (fᵏ - aᵏ⁻¹ * ϕ[i-1, j, k]) / β
            definitely_diagonally_dominant && (ϕ[i, j, k] = ϕ★)
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
            # the algorithm is unstable and we elide the forward pass update of ϕ.
            definitely_diagonally_dominant = abs(β) > 10 * eps(float_eltype(ϕ))
            ϕ★ = (fᵏ - aᵏ⁻¹ * ϕ[i, j-1, k]) / β
            definitely_diagonally_dominant && (ϕ[i, j, k] = ϕ★)
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

# The `ZDirection` sweep. The operations on values are those of the textbook loop
#
#     ϕ[1] = f[1] / β;  for k = 2:Nz:  t[k] = c[k-1] / β;  β = b[k] - a[k-1] t[k];  ϕ[k] = (f[k] - a[k-1] ϕ[k-1]) / β
#     for k = Nz-1:-1:1:  ϕ[k] -= t[k+1] ϕ[k+1]
#
# (with the masked products and the `β ≈ 0` fallback), in the same order and in the same types. What differs is where
# a value comes from, because on a GPU every thread walks a column and waits on each load in turn:
#
# - `ϕ[k-1]` in the forward sweep and `ϕ[k+1]` in the back-substitution are the values just stored, converted to the
#   element type of `ϕ` as the store converts them, instead of a load that follows the store;
# - the back-substitution takes four levels at a time: the eight loads of `t` and `ϕ` first, then the four dependent
#   updates, then the four stores;
# - a right-hand side that is a three-dimensional array is loaded one level ahead (`right_hand_side_ahead`).
#
# This relies on the scratch `t` sharing no memory with `ϕ` or with the right-hand side (it is the solver's own
# array), and on a right-hand side array being `ϕ` itself or an array the sweep does not write: then entry `k + 1`
# is untouched while row `k` is written.
@inline function solve_batched_tridiagonal_system_z!(i, j, Nz, ϕ, a, b, c, f, t, grid, p, args, tridiagonal_direction)
    FT = eltype(ϕ)
    TT = eltype(t)
    ahead = right_hand_side_ahead(f)
    ε = 10 * eps(float_eltype(ϕ))

    @inbounds begin
        β  = get_coefficient(i, j, 1, grid, b, p, tridiagonal_direction, args...)
        f₁ = get_coefficient(i, j, 1, grid, f, p, tridiagonal_direction, args...)
        ϕᵏ⁻¹ = convert(FT, f₁ / β)
        ϕ[i, j, 1] = ϕᵏ⁻¹

        fᵏ = ahead ? get_coefficient(i, j, min(2, Nz), grid, f, p, tridiagonal_direction, args...) : f₁

        for k = 2:Nz
            # the right-hand side of the next row (of this one again at k = Nz, unused)
            fᵏ⁺¹ = ahead ? get_coefficient(i, j, ifelse(k < Nz, k+1, k), grid, f, p, tridiagonal_direction, args...) : fᵏ

            cᵏ⁻¹ = get_coefficient(i, j, k-1, grid, c, p, tridiagonal_direction, args...)
            bᵏ   = get_coefficient(i, j, k,   grid, b, p, tridiagonal_direction, args...)
            aᵏ⁻¹ = get_coefficient(i, j, k-1, grid, a, p, tridiagonal_direction, args...)

            tᵏ = convert(TT, cᵏ⁻¹ / β)
            t[i, j, k] = tᵏ
            β = bᵏ - masked_multiply(aᵏ⁻¹, tᵏ)
            fᵏ = ahead ? fᵏ : get_coefficient(i, j, k, grid, f, p, tridiagonal_direction, args...)

            # If the problem is not diagonally-dominant such that `β ≈ 0`,
            # the algorithm is unstable and we elide the forward pass update of `ϕ`
            # (`ϕ[k]` is then neither stored nor, unless needed, loaded).
            definitely_diagonally_dominant = abs(β) > ε
            ϕ★ = (fᵏ - masked_multiply(aᵏ⁻¹, ϕᵏ⁻¹)) / β
            if definitely_diagonally_dominant
                ϕᵏ = convert(FT, ϕ★)
                ϕ[i, j, k] = ϕᵏ
            else
                ϕᵏ = ϕ[i, j, k]
            end

            ϕᵏ⁻¹ = ϕᵏ
            fᵏ = fᵏ⁺¹
        end

        back_substitute_z!(ϕ, t, i, j, Nz, ϕᵏ⁻¹)
    end

    return nothing
end

@inline right_hand_side_ahead(f::AbstractArray{<:Any, 3}) = true
@inline right_hand_side_ahead(f) = false

# ϕ[k] -= t[k+1] ϕ[k+1] for k = Nz-1, …, 1, with `ϕᴺ` the value of `ϕ[i, j, Nz]`.
@inline function back_substitute_z!(ϕ, t, i, j, Nz, ϕᴺ)
    FT = eltype(ϕ)
    ϕᵏ⁺¹ = ϕᴺ
    k = Nz - 1

    @inbounds while k ≥ 4
        t₁ = t[i, j, k+1]
        t₂ = t[i, j, k]
        t₃ = t[i, j, k-1]
        t₄ = t[i, j, k-2]
        ϕ₁ = ϕ[i, j, k]
        ϕ₂ = ϕ[i, j, k-1]
        ϕ₃ = ϕ[i, j, k-2]
        ϕ₄ = ϕ[i, j, k-3]

        ϕ₁ = convert(FT, ϕ₁ - masked_multiply(t₁, ϕᵏ⁺¹))
        ϕ₂ = convert(FT, ϕ₂ - masked_multiply(t₂, ϕ₁))
        ϕ₃ = convert(FT, ϕ₃ - masked_multiply(t₃, ϕ₂))
        ϕ₄ = convert(FT, ϕ₄ - masked_multiply(t₄, ϕ₃))

        ϕ[i, j, k]   = ϕ₁
        ϕ[i, j, k-1] = ϕ₂
        ϕ[i, j, k-2] = ϕ₃
        ϕ[i, j, k-3] = ϕ₄

        ϕᵏ⁺¹ = ϕ₄
        k -= 4
    end

    @inbounds while k ≥ 1
        ϕᵏ = convert(FT, ϕ[i, j, k] - masked_multiply(t[i, j, k+1], ϕᵏ⁺¹))
        ϕ[i, j, k] = ϕᵏ
        ϕᵏ⁺¹ = ϕᵏ
        k -= 1
    end

    return nothing
end

# Launched 32 × 8 rather than 16 × 16 (where the launch supports a per-kernel configuration): each thread sweeps
# one line, and more of a block's lines then lie along the first launch dimension, contiguous in memory when it
# is x (measured for the `ZDirection` solve over `:xy`). Applies to any two-dimensional configured workgroup
# (the configured default is (32, 4) on this branch).
Utils.kernel_launch_workgroup(::typeof(Utils.gpu_kernel_function(solve_batched_tridiagonal_system_kernel!)), ::Val{<:NTuple{2, Int}}) = (32, 8)
