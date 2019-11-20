struct BatchedTridiagonalSolver{A, B, C, F, T}
     a :: A
     b :: B
     c :: C
     f :: F
     t :: T
    Nx :: Int
    Ny :: Int
    Nz :: Int
end

function BatchedTridiagonalSolver(; dl, d, du, f, size)
    Nx, Ny, Nz = size
    t = zeros(Nz)
    return BatchedTridiagonalSolver(dl, d, du, f, t, Nx, Ny, Nz)
end

@inline get_coefficient(a::AbstractArray{T, 1}, i, j, k) where {T} = @inbounds a[k]
@inline get_coefficient(a::AbstractArray{T, 3}, i, j, k) where {T} = @inbounds a[i, j, k]
@inline get_coefficient(a::Function, i, j, k) = a(i, j, k)

function solve_batched_tridiagonal_system!(ϕ, solver)
    for j = 1:solver.Ny, i = 1:solver.Nx
        solve_tridiagonal_system!(ϕ, solver, i, j)
    end
end

"""
Solve the tridiagonal system of linear equations described by the tridiagonal
matrix `M` with right-hand-side `f` assuming one of the eigenvalues is zero
(which results in a singular matrix so the general Thomas algorithm has been
modified slightly).

Reference CPU implementation per Numerical Recipes, Press et. al 1992 (§ 2.4).
"""
function solve_tridiagonal_system!(ϕ, solver, i, j)
    a, b, c, f, t, N = solver.a, solver.b, solver.c, solver.f, solver.t, solver.Nz

    @inbounds begin
        β  = get_coefficient(b, i, j, 1)
        f₁ = get_coefficient(f, i, j, 1)
        ϕ[i, j, 1] = f₁ / β

        for k = 2:N
            cₖ₋₁ = get_coefficient(c, i, j, k-1)
            bₖ   = get_coefficient(b, i, j, k)
            aₖ₋₁ = get_coefficient(a, i, j, k-1)

            t[k] = cₖ₋₁ / β
            β    = bₖ - aₖ₋₁ * t[k]

            # This should only happen on last element of forward pass for problems
            # with zero eigenvalue. In that case the algorithmn is still stable.
            abs(β) < 1e-12 && break

            fₖ = get_coefficient(f, i, j, k)

            ϕ[i, j, k] = (fₖ - aₖ₋₁ * ϕ[i, j, k-1]) / β
        end

        for k = N-1:-1:1
            ϕ[i, j, k] = ϕ[i, j, k] - t[k+1] * ϕ[k+1]
        end
    end

    return nothing
end
