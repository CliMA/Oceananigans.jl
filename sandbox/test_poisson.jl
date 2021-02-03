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
    inds = 1:N
    λ = CuArray(@. (2sin((inds - 1) * π / N) / (L / N))^2)

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

@show test_poisson_1d_periodic(16, 1)
@show test_poisson_1d_periodic(17, rand())
