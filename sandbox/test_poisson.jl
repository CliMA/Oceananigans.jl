#####
##### Solving ∇²ϕ = R
#####

using Test
using FFTW
using KernelAbstractions
using Oceananigans
using Oceananigans.Architectures
using Oceananigans.Operators
using Oceananigans.Utils
using Oceananigans.BoundaryConditions: fill_halo_regions!

import CUDA
using CUDA: CuArray

CUDA.allowscalar(true)

#####
##### Makhoul DCT
#####

"""
    ω(M, k)
Return the `M`th root of unity raised to the `k`th power.
"""
@inline ω(M, k) = exp(-2im*π*k/M)

@inline permute(i, N) = isodd(i) ? floor(Int, i/2) + 1 : N - floor(Int, (i-1)/2)

@inline unpermute(i, N) = i <= ceil(N/2) ? 2i-1 : 2(N-i+1)

function dct_makhoul_1d(A::CuArray)
    B = similar(A)
    N = length(A)

    for k in 1:N
        B[permute(k, N)] = A[k]
    end

    B = CUDA.CUFFT.fft(B)

    for k in 1:N
        B[k] = 2 * ω(4N, k-1) * B[k]
    end

    return real(B)
end

function idct_makhoul_1d(A::CuArray)
    B = similar(A, complex(eltype(A)))
    N = length(A)

    B[1] = 1/2 * ω(4N, 0) * A[1]
    for k in 2:N
        B[k] = ω(4N, 1-k) * A[k]
    end

    B = CUDA.CUFFT.ifft(B)

    C = similar(A)
    for k in 1:N
        C[unpermute(k, N)] = real(B[k])
    end

    return C
end

function dct_makhoul_2d(A::CuArray)
    Nx, Ny = size(A)

    # DCT along dimension 1

    B = similar(A)

    for j in 1:Ny, i in 1:Nx
        B[permute(i, Nx), j] = A[i, j]
    end

    B = CUDA.CUFFT.fft(B, 1)

    for j in 1:Ny, i in 1:Nx
        B[i, j] = 2 * ω(4Nx, i-1) * B[i, j]
    end

    B = real(B)

    # DCT along dimension 2

    C = similar(A)

    for j in 1:Ny, i in 1:Nx
        C[i, permute(j, Ny)] = B[i, j]
    end

    C = CUDA.CUFFT.fft(C, 2)

    for j in 1:Ny, i in 1:Nx
        C[i, j] = 2 * ω(4Ny, j-1) * C[i, j]
    end

    return real(C)
end

function idct_makhoul_2d(A::CuArray)
    Nx, Ny = size(A)

    # IDCT along dimension 1

    B = similar(A, complex(eltype(A)))

    for j in 1:Ny
        B[1, j] = 1/2 * ω(4Nx, 0) * A[1, j]
    end

    for j in 1:Ny, i in 2:Nx
        B[i, j] = ω(4Nx, 1-i) * A[i, j]
    end

    B = CUDA.CUFFT.ifft(B, 1)

    C = similar(A)
    for j in 1:Ny, i in 1:Nx
        C[unpermute(i, Nx), j] = real(B[i, j])
    end

    # IDCT along dimension 2

    D = similar(A, complex(eltype(A)))

    for i in 1:Nx
        D[i, 1] = 1/2 * ω(4Ny, 0) * C[i, 1]
    end

    for j in 2:Ny, i in 1:Nx
        D[i, j] = ω(4Ny, 1-j) * C[i, j]
    end

    D = CUDA.CUFFT.ifft(D, 2)

    E = similar(A)
    for j in 1:Ny, i in 1:Nx
        E[i, unpermute(j, Ny)] = real(D[i, j])
    end

    return E
end

#####
##### Utils
#####

reshaped_size(N, dim) = dim == 1 ? (N, 1, 1) :
                        dim == 2 ? (1, N, 1) :
                        dim == 3 ? (1, 1, N) : nothing

"""
    poisson_eigenvalues(N, L, dim, ::Periodic)

Return the eigenvalues satisfying the discrete form of Poisson's equation
with periodic boundary conditions along the dimension `dim` with `N` grid
points and domain extent `L`.
"""
function poisson_eigenvalues(N, L, dim, ::Periodic)
    inds = reshape(1:N, reshaped_size(N, dim)...)
    return @. (2sin((inds - 1) * π / N) / (L / N))^2
end

"""
    poisson_eigenvalues(N, L, dim, ::Bounded)

Return the eigenvalues satisfying the discrete form of Poisson's equation
with staggered Neumann boundary conditions along the dimension `dim` with
`N` grid points and domain extent `L`.
"""
function poisson_eigenvalues(N, L, dim, ::Bounded)
    inds = reshape(1:N, reshaped_size(N, dim)...)
    return @. (2sin((inds - 1) * π / 2N) / (L / N))^2
end

@kernel function ∇²!(grid, f, ∇²f)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²f[i, j, k] = ∇²(i, j, k, grid, f)
end

@kernel function divergence!(grid, u, v, w, div)
    i, j, k = @index(Global, NTuple)
    @inbounds div[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
end

function random_divergent_source_term(FT, arch, grid)
    # Generate right hand side from a random (divergent) velocity field.
    Ru = CellField(FT, arch, grid, UVelocityBoundaryConditions(grid))
    Rv = CellField(FT, arch, grid, VVelocityBoundaryConditions(grid))
    Rw = CellField(FT, arch, grid, WVelocityBoundaryConditions(grid))
    U = (u=Ru, v=Rv, w=Rw)

    Nx, Ny, Nz = size(grid)
    set!(Ru, rand(Nx, Ny, Nz))
    set!(Rv, rand(Nx, Ny, Nz))
    set!(Rw, rand(Nx, Ny, Nz))

    # Adding (nothing, nothing) in case we need to dispatch on ::NFBC
    fill_halo_regions!(Ru, arch, nothing, nothing)
    fill_halo_regions!(Rv, arch, nothing, nothing)
    fill_halo_regions!(Rw, arch, nothing, nothing)

    # Compute the right hand side R = ∇⋅U
    ArrayType = array_type(arch)
    R = zeros(Nx, Ny, Nz) |> ArrayType
    event = launch!(arch, grid, :xyz, divergence!, grid, U.u.data, U.v.data, U.w.data, R,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    return R
end

function compute_∇²(ϕ_array, FT, arch, grid)
    p_bcs = PressureBoundaryConditions(grid)
    ϕ = CellField(FT, arch, grid, p_bcs)  # "pressure"
    ∇²ϕ = CellField(FT, arch, grid, p_bcs)

    set!(ϕ, ϕ_array)

    fill_halo_regions!(ϕ, arch)
    event = launch!(arch, grid, :xyz, ∇²!, grid, ϕ.data, ∇²ϕ.data, dependencies=Event(device(arch)))
    wait(device(arch), event)
    fill_halo_regions!(∇²ϕ, arch)

    return interior(∇²ϕ)
end

#####
##### 1D Periodic
#####

function solve_poisson_1d_periodic(R::CuArray, L)
    N = length(R)
    λ = poisson_eigenvalues(N, L, 1, Periodic()) |> CuArray
    λ = reshape(λ, N)

    ϕ = CUDA.CUFFT.fft(R)

    @. ϕ = -ϕ/λ
    ϕ[1] = 0

    ϕ = CUDA.CUFFT.ifft(ϕ)

    return ϕ
end

function test_poisson_1d_periodic(N, L)
    topo = (Periodic, Periodic, Periodic)
    grid = RegularCartesianGrid(topology=topo, size=(N, 1, 1), extent=(L, 1, 1))
    RHS = random_divergent_source_term(Float64, Oceananigans.GPU(), grid)
    RHS = reshape(RHS, N)
    ϕ = solve_poisson_1d_periodic(RHS, L) |> real
    ∇²ϕ = compute_∇²(ϕ, Float64, Oceananigans.GPU(), grid)
    return @test ∇²ϕ ≈ RHS
end

@testset "1D Periodic" begin
    @show test_poisson_1d_periodic(16, 1)
    @show test_poisson_1d_periodic(17, rand())
end

#####
##### 1D Bounded
#####

function solve_poisson_1d_bounded(R::CuArray, L)
    N = length(R)
    λ = poisson_eigenvalues(N, L, 1, Bounded()) |> CuArray
    λ = reshape(λ, N)

    ϕ = dct_makhoul_1d(R)

    @. ϕ = -ϕ/λ
    ϕ[1] = 0

    ϕ = idct_makhoul_1d(ϕ)

    return ϕ
end

function test_poisson_1d_bounded(N, L)
    topo = (Bounded, Periodic, Periodic)
    grid = RegularCartesianGrid(topology=topo, size=(N, 1, 1), extent=(L, 1, 1))
    RHS = random_divergent_source_term(Float64, Oceananigans.GPU(), grid)
    RHS = reshape(RHS, N)
    ϕ = solve_poisson_1d_bounded(RHS, L) |> real
    ∇²ϕ = compute_∇²(ϕ, Float64, Oceananigans.GPU(), grid)
    return @test ∇²ϕ ≈ RHS
end

@testset "1D Bounded" begin
    @show test_poisson_1d_bounded(16, 1)
    @show test_poisson_1d_bounded(17, rand())
end

#####
##### 2D Periodic
#####

function solve_poisson_2d_periodic(R::CuArray, Lx, Ly)
    Nx, Ny = size(R)
    λx = poisson_eigenvalues(Nx, Lx, 1, Periodic()) |> CuArray
    λy = poisson_eigenvalues(Ny, Ly, 2, Periodic()) |> CuArray
    λx = reshape(λx, Nx, 1)
    λy = reshape(λy, 1, Ny)

    ϕ = CUDA.CUFFT.fft(R)

    @. ϕ = - ϕ / (λx + λy)
    ϕ[1, 1] = 0

    ϕ = CUDA.CUFFT.ifft(ϕ)

    return ϕ
end

function test_poisson_2d_periodic(Nx, Ny, Lx, Ly)
    topo = (Periodic, Periodic, Periodic)
    grid = RegularCartesianGrid(topology=topo, size=(Nx, Ny, 1), extent=(Lx, Ly, 1))
    RHS = random_divergent_source_term(Float64, Oceananigans.GPU(), grid)
    RHS = reshape(RHS, Nx, Ny)
    ϕ = solve_poisson_2d_periodic(RHS, Lx, Ly) |> real
    ∇²ϕ = compute_∇²(ϕ, Float64, Oceananigans.GPU(), grid)
    return @test ∇²ϕ ≈ RHS
end

@testset "2D (Periodic, Periodic)" begin
    @show test_poisson_2d_periodic(16, 16, 1, 1)
    @show test_poisson_2d_periodic(16, 32, 1, 2)
    @show test_poisson_2d_periodic(32, 16, rand(), rand())
    @show test_poisson_2d_periodic(32, 32, rand(), rand())
    @show test_poisson_2d_periodic(11, 17, rand(), rand())
    @show test_poisson_2d_periodic(23, 19, rand(), rand())
end

#####
##### 2D Bounded
#####

function solve_poisson_2d_bounded(R::CuArray, Lx, Ly)
    Nx, Ny = size(R)
    λx = poisson_eigenvalues(Nx, Lx, 1, Bounded()) |> CuArray
    λy = poisson_eigenvalues(Ny, Ly, 2, Bounded()) |> CuArray
    λx = reshape(λx, Nx, 1)
    λy = reshape(λy, 1, Ny)

    ϕ = dct_makhoul_2d(R)

    @. ϕ = - ϕ / (λx + λy)
    ϕ[1, 1] = 0

    ϕ = idct_makhoul_2d(ϕ)

    return ϕ
end

function test_poisson_2d_bounded(Nx, Ny, Lx, Ly)
    topo = (Bounded, Bounded, Periodic)
    grid = RegularCartesianGrid(topology=topo, size=(Nx, Ny, 1), extent=(Lx, Ly, 1))
    RHS = random_divergent_source_term(Float64, Oceananigans.GPU(), grid)
    RHS = reshape(RHS, Nx, Ny)
    ϕ = solve_poisson_2d_bounded(RHS, Lx, Ly) |> real
    ∇²ϕ = compute_∇²(ϕ, Float64, Oceananigans.GPU(), grid)
    return @test ∇²ϕ ≈ RHS
end

@testset "2D (Bounded, Bounded)" begin
    @show test_poisson_2d_bounded(16, 16, 1, 1)
    @show test_poisson_2d_bounded(16, 32, 0.2, 1.7)
    @show test_poisson_2d_bounded(32, 16, rand(), rand())
    @show test_poisson_2d_bounded(32, 32, rand(), rand())
    @show test_poisson_2d_bounded(11, 17, rand(), rand())
    @show test_poisson_2d_bounded(23, 19, rand(), rand())
end

#####
##### 2D (Bounded, Periodic)
#####

function dct_makhoul_2d_dim1(A::CuArray)
    B = similar(A)
    Nx, Ny = size(A)

    for i in 1:Nx, j in 1:Ny
        B[permute(i, Nx), j] = A[i, j]
    end

    B = CUDA.CUFFT.fft(B, 1)

    for i in 1:Nx, j in 1:Ny
        B[i, j] = 2 * ω(4Nx, i-1) * B[i, j]
    end

    return real(B)
end

function idct_makhoul_2d_dim1(A::CuArray)
    B = similar(A, complex(eltype(A)))
    Nx, Ny = size(A)

    for j in 1:Ny
        B[1, j] = 1/2 * ω(4Nx, 0) * A[1, j]
        for i in 2:Nx
            B[i, j] = ω(4Nx, 1-i) * A[i, j]
        end
    end

    B = CUDA.CUFFT.ifft(B, 1)

    C = similar(A)
    for i in 1:Nx, j in 1:Ny
        C[unpermute(i, Nx), j] = real(B[i, j])
    end

    return C
end

function solve_poisson_2d_bp(R::CuArray, Lx, Ly)
    Nx, Ny = size(R)
    λx = poisson_eigenvalues(Nx, Lx, 1, Bounded()) |> CuArray
    λy = poisson_eigenvalues(Ny, Ly, 2, Periodic()) |> CuArray
    λx = reshape(λx, Nx, 1)
    λy = reshape(λy, 1, Ny)

    ϕ = dct_makhoul_2d_dim1(R)
    ϕ = CUDA.CUFFT.fft(ϕ, 2)

    @. ϕ = - ϕ / (λx + λy)
    ϕ[1, 1] = 0

    ϕ = CUDA.CUFFT.ifft(ϕ, 2)
    ϕ = idct_makhoul_2d_dim1(ϕ)

    return ϕ
end

function test_poisson_2d_bp(Nx, Ny, Lx, Ly)
    topo = (Bounded, Periodic, Periodic)
    grid = RegularCartesianGrid(topology=topo, size=(Nx, Ny, 1), extent=(Lx, Ly, 1))
    RHS = random_divergent_source_term(Float64, Oceananigans.GPU(), grid)
    RHS = reshape(RHS, Nx, Ny)
    ϕ = solve_poisson_2d_bp(RHS, Lx, Ly) |> real
    ∇²ϕ = compute_∇²(ϕ, Float64, Oceananigans.GPU(), grid)
    return @test ∇²ϕ ≈ RHS
end

@testset "2D (Bounded, Periodic)" begin
    @show test_poisson_2d_bp(16, 16, 1, 1)
    @show test_poisson_2d_bp(16, 32, 0.2, 1.7)
    @show test_poisson_2d_bp(32, 16, rand(), rand())
    @show test_poisson_2d_bp(32, 32, rand(), rand())
    @show test_poisson_2d_bp(11, 17, rand(), rand())
    @show test_poisson_2d_bp(23, 19, rand(), rand())
end
