using Oceananigans
using LinearAlgebra, Random, Statistics, FFTW, Test

function λi(Nx, Δx)
    is = reshape(1:Nx, Nx, 1, 1)
    @. (2sin((is-1)*π/Nx) / Δx)^2
end

function λj(Ny, Δy)
    js = reshape(1:Ny, 1, Ny, 1)
    @. (2sin((js-1)*π/Ny) / Δy)^2
end

"""
    tridiag(M::Tridiagonal{T,<:Array}, f::Vector{T})::Vector{T} where T
Solve the tridiagonal system of linear equations described by the tridiagonal
matrix `M` with right-hand-side `g` assuming one of the eigenvalues is zero
(which results in a singular matrix so the general Thomas algorithm has been
modified slightly).
Reference CPU implementation per Numerical Recipes, Press et. al 1992 (§ 2.4)
"""
function tridiag(M, f)
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

δ(k, ΔzF, ΔzC, kx², ky²) = - (1/ΔzF[k-1] + 1/ΔzF[k]) - ΔzC[k] * (kx² + ky²)

function solve_poisson_1d(Nz, ΔzF, ΔzC, kx², ky², F)
    ld = [1/ΔzF[k] for k in 1:Nz-1]
    ud = copy(ld)
    d = [-1/ΔzF[1], [δ(k, ΔzF, ΔzC, kx², ky²) for k in 2:Nz]...]    
    M = Tridiagonal(ld, d, ud)

    ϕ = tridiag(M, F)
    
    return ϕ
end

function grid(zF)
    Nz = length(zF) - 1
    ΔzF = [zF[k+1] - zF[k] for k in 1:Nz]
    zC = [(zF[k] + zF[k+1]) / 2 for k in 1:Nz]
    ΔzC = [zC[k+1] - zC[k] for k in 1:Nz-1]
    return zF, zC, ΔzF, ΔzC
end

Nx, Ny = 4, 4
Lx, Ly = 1, 1
Δx, Δy = Lx/Nx, Ly/Ny

zF = [1, 2, 4, 7, 11, 16, 22, 29, 37]
Nz = length(zF) - 1
zF, zC, ΔzF, ΔzC = grid(zF)

ΔzC = [ΔzC..., ΔzC[end]]

@show zF, zC, ΔzF, ΔzC

zC = reshape(zC, (1, 1, Nz))
zF = reshape(zF, (1, 1, Nz+1))
ΔzC = reshape(ΔzC, (1, 1, Nz))
ΔzF = reshape(ΔzF, (1, 1, Nz))

R = rand(MersenneTwister(0), Nx, Ny, Nz)
R .= R .- mean(R)

F = ΔzC .* R

F̃ = fft(F, [1, 2])
ϕ̃ = similar(F̃)

kx² = λi(Nx, Δx)
ky² = λj(Ny, Δy)

for i in 1:Nx, j in 1:Ny
    ϕ̃[i, j, :] = solve_poisson_1d(Nz, ΔzF, ΔzC, kx²[i], ky²[j], F̃[i, j, :])
end

# ϕ̃[1, 1, 1] = 0
ϕ = real.(ifft(ϕ̃, [1, 2]))
ϕ = ϕ .- mean(ϕ)

@inline δx_caa(i, j, k, f) = @inbounds f[i+1, j, k] - f[i, j, k]
@inline δy_aca(i, j, k, f) = @inbounds f[i, j+1, k] - f[i, j, k]
@inline δz_aac(i, j, k, f) = @inbounds f[i, j, k+1] - f[i, j, k]

@inline ∂x_caa(i, j, k, Δx,  f) = δx_caa(i, j, k, f) / Δx
@inline ∂y_aca(i, j, k, Δy,  f) = δy_aca(i, j, k, f) / Δy
@inline ∂z_aac(i, j, k, ΔzF, f) = δz_aac(i, j, k, f) / ΔzF[k]

@inline ∂x²(i, j, k, Δx, f)       = (∂x_caa(i, j, k, Δx, f)  - ∂x_caa(i-1, j, k, Δx, f))  / Δx
@inline ∂y²(i, j, k, Δy, f)       = (∂y_aca(i, j, k, Δy, f)  - ∂y_aca(i, j-1, k, Δy, f))  / Δy
@inline ∂z²(i, j, k, ΔzF, ΔzC, f) = (∂z_aac(i, j, k, ΔzF, f) - ∂z_aac(i, j, k-1, ΔzF, f)) / ΔzC[k]

@inline ∇²(i, j, k, Δx, Δy, ΔzF, ΔzC, f) = ∂x²(i, j, k, Δx, f) + ∂y²(i, j, k, Δy, f) + ∂z²(i, j, k, ΔzF, ΔzC, f)

∇²ϕ = zeros(Nx, Ny, Nz)

# ∇²ϕ[1] = ∂z_aac(1, ΔzF, ϕ) / ΔzC[1]
for i in 1:Nx, j in 1:Ny, k in 2:Nz-1
    ∇²ϕ[i, j, k] = ∇²(i, j, k, Δx, Δy, ΔzF, ΔzC, ϕ)
end
# ∇²ϕ[Nz] = (∂z_aac(Nz, ΔzF, ϕ) - ∂z_aac(Nz-1, ΔzF, ϕ)) / ΔzC[Nz-1]
∇²ϕ

@test ∇²ϕ[2:Nx-1, 2:Ny-1, 2:Nz-1] ≈ R[2:Nx-1, 2:Ny-1, 2:Nz-1]

