using Random, Statistics, LinearAlgebra, Test

using Oceananigans, Oceananigans.Operators

"""
    tridiag(M::Tridiagonal{T,<:Array}, f::Vector{T})::Vector{T} where T

Solve the tridiagonal system of linear equations described by the tridiagonal
matrix `M` with right-hand-side `g` assuming one of the eigenvalues is zero
(which results in a singular matrix so the general Thomas algorithm has been
modified slightly).

Reference CPU implementation per Numerical Recipes, Press et. al 1992 (§ 2.4)
"""
function tridiag(M::Tridiagonal{T,<:Array}, f::Vector{T})::Vector{T} where T
    N = length(f)
    ϕ = similar(f)
    γ = similar(f)

    β    = M.d[1]
    ϕ[1] = f[1] / β

    for j = 2:N
        γ[j] = M.du[j-1] / β
        β    = M.d[j] - M.dl[j-1] * γ[j]

        # This should only happen on last element of forward pass for problems
        # with zero eigenvalue. In that case the algorithmn is still stable.
        abs(β) < 1.0e-12 && break

        ϕ[j] = (f[j] - M.dl[j-1] * ϕ[j-1]) / β
    end

    for j = 1:N-1
        k = N-j
        ϕ[k] = ϕ[k] - γ[k+1] * ϕ[k+1]
    end

    return ϕ
end

function ∇²!(grid::RegularCartesianGrid, f, ∇²f)
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        @inbounds ∇²f[i, j, k] = ∇²(grid, f, i, j, k)
    end
end

Nz, Lz = 16, 1
Δz = (Lz/Nz) * ones(Nz+1)

RHS = rand(MersenneTwister(0), Nz+1)
RHS[Nz+1] = 0
RHS[1:Nz] .= RHS[1:Nz] .- mean(RHS[1:Nz])

# Diagonal elements.
δ(k) = -(Δz[k-1] + Δz[k+1]) / (Δz[k-1] * Δz[k+1])

ud = [1/Δz[k] for k in 2:Nz+1]
ld = [1/Δz[k] for k in 1:Nz]
d = [-1/Δz[1], [δ(k) for k in 2:Nz]..., -1/Δz[Nz+1]]

M = Tridiagonal(ld, d, ud)

g = Δz .* RHS # Must scale right-hand-side by Δz.

ψ = tridiag(M, g)

# Check that Laplacian of the solution matches the original RHS to machine ϵ.
grid = RegularCartesianGrid((1, 1, Nz), (1, 1, Lz))
fbcs = DoublyPeriodicBCs()

ϕ   = CellField(Float64, CPU(), grid)
∇²ϕ = CellField(Float64, CPU(), grid)

data(ϕ) .= real.(reshape(ψ[1:Nz], 1, 1, Nz))

fill_halo_regions!(grid, (:T, fbcs, ϕ.data))
∇²!(grid, ϕ, ∇²ϕ)

fill_halo_regions!(grid, (:T, fbcs, ∇²ϕ.data))

@test data(∇²ϕ) ≈ reshape(RHS[1:Nz], 1, 1, Nz)
