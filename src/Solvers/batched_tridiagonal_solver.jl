struct BatchedTridiagonalSolver{A, B, C, F, T, G, P}
       a :: A
       b :: B
       c :: C
       f :: F
       t :: T
    grid :: G
  params :: P
end

function BatchedTridiagonalSolver(; dl, d, du, f, grid, params=nothing)
    t = zeros(grid.Nz)
    return BatchedTridiagonalSolver(dl, d, du, f, t, grid, params)
end

@inline get_coefficient(a::AbstractArray{T, 1}, i, j, k, grid, p) where {T} = @inbounds a[k]
@inline get_coefficient(a::AbstractArray{T, 3}, i, j, k, grid, p) where {T} = @inbounds a[i, j, k]
@inline get_coefficient(a::Function, i, j, k, grid, p) = a(i, j, k, grid, p)

"""
Solve the tridiagonal system of linear equations described by the tridiagonal
matrix `M` with right-hand-side `f` assuming one of the eigenvalues is zero
(which results in a singular matrix so the general Thomas algorithm has been
modified slightly).

Reference CPU implementation per Numerical Recipes, Press et. al 1992 (§ 2.4).
"""
function solve_batched_tridiagonal_system!(ϕ, solver)
    grid = solver.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    a, b, c, f, t, p = solver.a, solver.b, solver.c, solver.f, solver.t, solver.params

    @inbounds begin
        for i = 1:Nx, j = 1:Ny
            β  = get_coefficient(b, i, j, 1, grid, p)
            f₁ = get_coefficient(f, i, j, 1, grid, p)
            ϕ[i, j, 1] = f₁ / β

            for k = 2:Nz
                cₖ₋₁ = get_coefficient(c, i, j, k-1, grid, p)
                bₖ   = get_coefficient(b, i, j, k,   grid, p)
                aₖ₋₁ = get_coefficient(a, i, j, k-1, grid, p)

                t[k] = cₖ₋₁ / β
                β    = bₖ - aₖ₋₁ * t[k]

                # This should only happen on last element of forward pass for problems
                # with zero eigenvalue. In that case the algorithmn is still stable.
                abs(β) < 1e-12 && break

                fₖ = get_coefficient(f, i, j, k, grid, p)

                ϕ[i, j, k] = (fₖ - aₖ₋₁ * ϕ[i, j, k-1]) / β
            end

            for k = Nz-1:-1:1
                ϕ[i, j, k] = ϕ[i, j, k] - t[k+1] * ϕ[i, j, k+1]
            end
        end
    end

    return nothing
end
