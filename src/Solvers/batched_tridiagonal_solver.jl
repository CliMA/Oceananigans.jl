"""
    BatchedTridiagonalSolver

A batched solver for large numbers of triadiagonal systems.
"""
struct BatchedTridiagonalSolver{A, B, C, F, T, G, P}
             a :: A
             b :: B
             c :: C
             f :: F
             t :: T
          grid :: G
    parameters :: P
end

"""
    BatchedTridiagonalSolver(; dl, d, du, f, grid, parameters=nothing)

Construct a solver for batched tridiagonal systems on `grid` of the form

                           bⁱʲ¹ ϕⁱʲ¹ + cⁱʲ²   ϕⁱʲ²   = fⁱʲ¹,  k = 1
           aⁱʲᵏ⁻¹ ϕⁱʲᵏ⁻¹ + bⁱʲᵏ ϕⁱʲᵏ + cⁱʲᵏ⁺¹ ϕⁱʲᵏ⁺¹ = fⁱʲᵏ,  k = 2, ..., N-1
           aⁱʲᴺ⁻¹ ϕⁱʲᴺ⁻¹ + bⁱʲᴺ ϕⁱʲᴺ                 = fⁱʲᴺ,  k = N

where `a` is the lower diagonal, `b` is the diagonal, `c` is the upper diagonal`, and `f` is the right hand side.

The coefficients `a`, `b`, `c`, and `f` can be specified in three ways:

1. A 1D array means that `aⁱʲᵏ = a[k]`.

2. A 3D array means that `aⁱʲᵏ = a[i, j, k]`.

3. Otherwise, `a` is assumed to be callable:
    * If `isnothing(parameters)` then `aⁱʲᵏ = a(i, j, k, grid)`.
    * If `!isnothing(parameters)` then `aⁱʲᵏ = a(i, j, k, grid, parameters)`.
"""
function BatchedTridiagonalSolver(arch=CPU(), FT=Float64; dl, d, du, f, grid, parameters=nothing)
    ArrayType = array_type(arch)
    t = zeros(FT, grid.Nx, grid.Ny, grid.Nz) |> ArrayType

    return BatchedTridiagonalSolver(dl, d, du, f, t, grid, parameters)
end

@inline get_coefficient(a::AbstractArray{T, 1}, i, j, k, grid, p) where {T} = @inbounds a[k]
@inline get_coefficient(a::AbstractArray{T, 3}, i, j, k, grid, p) where {T} = @inbounds a[i, j, k]
@inline get_coefficient(a, i, j, k, grid, p) = a(i, j, k, grid, p)
@inline get_coefficient(a, i, j, k, grid, ::Nothing) = a(i, j, k, grid)

"""
    solve_batched_tridiagonal_system!(ϕ, arch, solver)

Solve the batched tridiagonal system of linear equations described by the
`BatchedTridiagonalSolver` `solver` using a modified TriDiagonal Matrix Algorithm (TDMA)
that is still capable of solving singular systems with a zero eigenvalue.

The result is stored in `ϕ` which must have size `(grid.Nx, grid.Ny, grid.Nz)`.

Reference implementation per Numerical Recipes, Press et. al 1992 (§ 2.4).
"""
function solve_batched_tridiagonal_system!(ϕ, arch, solver; dependencies = Event(device(arch)))
    a, b, c, f, t, grid, parameters = solver.a, solver.b, solver.c, solver.f, solver.t, solver.grid, solver.parameters

    event = launch!(arch, grid, :xy,
                    solve_batched_tridiagonal_system_kernel!, ϕ, a, b, c, f, t, grid, parameters,
                    dependencies = dependencies)

    wait(device(arch), event)

    return nothing
end

float_eltype(ϕ::AbstractArray{T}) where T <: AbstractFloat = T
float_eltype(ϕ::AbstractArray{<:Complex{T}}) where T <: AbstractFloat = T

@kernel function solve_batched_tridiagonal_system_kernel!(ϕ, a, b, c, f, t, grid, p)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    i, j = @index(Global, NTuple)

    @inbounds begin
        β  = get_coefficient(b, i, j, 1, grid, p)
        f₁ = get_coefficient(f, i, j, 1, grid, p)
        ϕ[i, j, 1] = f₁ / β

        @unroll for k = 2:Nz
            cᵏ⁻¹ = get_coefficient(c, i, j, k-1, grid, p)
            bᵏ   = get_coefficient(b, i, j, k,   grid, p)
            aᵏ⁻¹ = get_coefficient(a, i, j, k-1, grid, p)

            t[i, j, k] = cᵏ⁻¹ / β
            β = bᵏ- aᵏ⁻¹ * t[i, j, k]

            fᵏ = get_coefficient(f, i, j, k, grid, p)

            # This should only happen on last element of forward pass for problems
            # with zero eigenvalue. In that case the algorithmn is still stable.
            if abs(β) < 1000 * eps(float_eltype(ϕ))
                ϕ[i, j, k] = 0
            else
                ϕ[i, j, k] = (fᵏ- aᵏ⁻¹ * ϕ[i, j, k-1]) / β
            end
        end

        @unroll for k = Nz-1:-1:1
            ϕ[i, j, k] = ϕ[i, j, k] - t[i, j, k+1] * ϕ[i, j, k+1]
        end
    end
end
