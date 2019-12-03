@hascuda using CUDAnative
using GPUifyLoops: @launch, @loop, @unroll

using Oceananigans: @loop_xy, device, array_type, launch_config

"""
    BatchedTridiagonalSolver

A solver for batched tridiagonal systems.
"""
struct BatchedTridiagonalSolver{A, B, C, F, T, G, P}
       a :: A
       b :: B
       c :: C
       f :: F
       t :: T
    grid :: G
  params :: P
end

"""
    BatchedTridiagonalSolver(; dl, d, du, f, grid, params=nothing)

Construct a solver for batched tridiagonal systems of the form

                               b(i, j, 1)ϕ(i, j, 1) + c(i, j,   2)ϕ(i, j,   2) = f(i, j, 1),  k = 1
    a(i, j, k-1)ϕ(i, j, k-1) + b(i, j, k)ϕ(i, j, k) + c(i, j, k+1)ϕ(i, j, k+1) = f(i, j, k),  k = 2, ..., N-1
    a(i, j, N-1)ϕ(i, j, N-1) + b(i, j, N)ϕ(i, j, N)                            = f(i, j, N),  k = N

where `dl` stores the lower diagonal coefficients `a(i, j, k)`, `d` stores the diagonal coefficients `b(i, j, k)`,
`du` stores the upper diagonal coefficients `c(i, j, k)`, and `f` stores the right-hand-side terms `f(i, j, k)`. A
`grid` must be passed in.

`dl`, `d`, `du`, and `f` can be specified in three ways to describe different batched systems:
1. A 1D array indicates that the coefficients only depend on `k` and are the same for all the tridiagonal systems, i.e.
   `a(i, j, k) = a(k)`.
2. A 3D array indicates that the coefficients are different for each tridiagonal systems and depend on `(i, j, k)`.
3. A function with the signature `b(i, j, k, grid, params)` that returns the coefficient `b(i, j, k)`.

`params` is an optional named tuple of parameters that is accessible to functions.
"""
function BatchedTridiagonalSolver(arch=CPU(); dl, d, du, f, grid, params=nothing)
    ArrayType = array_type(arch)
    t = zeros(grid.Nx, grid.Ny, grid.Nz) |> ArrayType

    return BatchedTridiagonalSolver(dl, d, du, f, t, grid, params)
end

@inline get_coefficient(a::AbstractArray{T, 1}, i, j, k, grid, p) where {T} = @inbounds a[k]
@inline get_coefficient(a::AbstractArray{T, 3}, i, j, k, grid, p) where {T} = @inbounds a[i, j, k]
@inline get_coefficient(a::Function, i, j, k, grid, p) = a(i, j, k, grid, p)

"""
Solve the batched tridiagonal system of linear equations described by the
`BatchedTridiagonalSolver` `solver` using a modified TriDiagonal Matrix Algorithm (TDMA)
that is still capable of solving singular systems with a zero eigenvalue.

The result is stored in `ϕ` which must have size `(grid.Nx, grid.Ny, grid.Nz)`.

Reference implementation per Numerical Recipes, Press et. al 1992 (§ 2.4).
"""
function solve_batched_tridiagonal_system!(ϕ, arch, solver)
    a, b, c, f, t, grid, params = solver.a, solver.b, solver.c, solver.f, solver.t, solver.grid, solver.params

    @launch(device(arch), config=launch_config(grid, 2),
            solve_batched_tridiagonal_system_kernel!(ϕ, a, b, c, f, t, grid, params))

    return nothing
end

function solve_batched_tridiagonal_system_kernel!(ϕ, a, b, c, f, t, grid, p)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    @loop_xy i j grid begin
        @inbounds begin
            β  = get_coefficient(b, i, j, 1, grid, p)
            f₁ = get_coefficient(f, i, j, 1, grid, p)
            ϕ[i, j, 1] = f₁ / β

            @unroll for k = 2:Nz
                cₖ₋₁ = get_coefficient(c, i, j, k-1, grid, p)
                bₖ   = get_coefficient(b, i, j, k,   grid, p)
                aₖ₋₁ = get_coefficient(a, i, j, k-1, grid, p)

                t[i, j, k] = cₖ₋₁ / β
                β    = bₖ - aₖ₋₁ * t[i, j, k]

                # This should only happen on last element of forward pass for problems
                # with zero eigenvalue. In that case the algorithmn is still stable.
                abs(β) < 1e-12 && break

                fₖ = get_coefficient(f, i, j, k, grid, p)

                ϕ[i, j, k] = (fₖ - aₖ₋₁ * ϕ[i, j, k-1]) / β
            end

            @unroll for k = Nz-1:-1:1
                ϕ[i, j, k] = ϕ[i, j, k] - t[i, j, k+1] * ϕ[i, j, k+1]
            end
        end
    end

    return nothing
end
