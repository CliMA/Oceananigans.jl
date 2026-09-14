include("dependencies_for_runtests.jl")

using LinearAlgebra
using Oceananigans.Architectures: array_type
using Oceananigans.Grids: XDirection, YDirection, ZDirection

import Adapt: adapt_structure
import Oceananigans.Solvers: get_coefficient, first_row

# A diagonal whose system spans only the rows `first_rows[i, j]:Nz` of each column
struct PartialColumnDiagonal{B, K}
    b :: B
    first_rows :: K
end

adapt_structure(to, d::PartialColumnDiagonal) = PartialColumnDiagonal(adapt(to, d.b), adapt(to, d.first_rows))

@inline get_coefficient(i, j, k, grid, d::PartialColumnDiagonal, p, ::ZDirection, args...) = @inbounds d.b[k]
@inline first_row(i, j, grid, d::PartialColumnDiagonal, p, ::ZDirection, args...) = @inbounds d.first_rows[i, j]

function can_solve_single_tridiagonal_system(arch, N; tridiagonal_direction=ZDirection())
    ArrayType = array_type(arch)

    a = rand(N-1)
    b = 3 .+ rand(N) # +3 to ensure diagonal dominance.
    c = rand(N-1)
    f = rand(N)

    # Solve the system with backslash on the CPU to avoid scalar operations on the GPU.
    M = Tridiagonal(a, b, c)
    ϕ_correct = M \ f

    # Convert to CuArray if needed.
    a, b, c, f = ArrayType.([a, b, c, f])

    if tridiagonal_direction isa XDirection
        ϕ = reshape(zeros(N), (N, 1, 1)) |> ArrayType
        grid = RectilinearGrid(arch, size=(N, 1, 1), extent=(1, 1, 1))
    elseif tridiagonal_direction isa YDirection
        ϕ = reshape(zeros(N), (1, N, 1)) |> ArrayType
        grid = RectilinearGrid(arch, size=(1, N, 1), extent=(1, 1, 1))
    elseif tridiagonal_direction isa ZDirection
        ϕ = reshape(zeros(N), (1, 1, N)) |> ArrayType
        grid = RectilinearGrid(arch, size=(1, 1, N), extent=(1, 1, 1))
    end

    btsolver = BatchedTridiagonalSolver(grid;
                                        lower_diagonal = a,
                                        diagonal = b,
                                        upper_diagonal = c,
                                        tridiagonal_direction)

    solve!(ϕ, btsolver, f)

    return Array(ϕ[:]) ≈ ϕ_correct
end

function can_solve_batched_tridiagonal_system_with_3D_RHS(arch, Nx, Ny, Nz; tridiagonal_direction = ZDirection())
    ArrayType = array_type(arch)

    N = if tridiagonal_direction isa XDirection
            Nx
        elseif tridiagonal_direction == YDirection()
            Ny
        elseif tridiagonal_direction isa ZDirection
            Nz
        end

    a = rand(N-1)
    b = 3 .+ rand(N) # +3 to ensure diagonal dominance.
    c = rand(N-1)
    f = rand(Nx, Ny, Nz)

    M = Tridiagonal(a, b, c)
    ϕ_correct = zeros(Nx, Ny, Nz)

    # Solve the systems with backslash on the CPU to avoid scalar operations on the GPU.
    if tridiagonal_direction isa XDirection
        for j = 1:Ny, k = 1:Nz
            ϕ_correct[:, j, k] .= M \ f[:, j, k]
        end
    elseif tridiagonal_direction isa YDirection
        for i = 1:Nx, k = 1:Nz
            ϕ_correct[i, :, k] .= M \ f[i, :, k]
        end
    elseif tridiagonal_direction isa ZDirection
        for i = 1:Nx, j = 1:Ny
            ϕ_correct[i, j, :] .= M \ f[i, j, :]
        end
    end

    # Convert to CuArray if needed.
    a, b, c, f = ArrayType.([a, b, c, f])

    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(1, 1, 1))
    btsolver = BatchedTridiagonalSolver(grid;
                                        lower_diagonal = a,
                                        diagonal = b,
                                        upper_diagonal = c,
                                        tridiagonal_direction)

    ϕ = zeros(Nx, Ny, Nz) |> ArrayType

    solve!(ϕ, btsolver, f)

    return Array(ϕ) ≈ ϕ_correct
end

function can_solve_partial_columns(arch, Nx, Ny, Nz)
    ArrayType = array_type(arch)

    a = rand(Nz-1)
    b = 3 .+ rand(Nz) # +3 to ensure diagonal dominance.
    c = rand(Nz-1)
    f = rand(Nx, Ny, Nz)

    # Every column starts at a different row, including columns that are skipped altogether
    first_rows = [mod(i - 1 + Nx * (j - 1), Nz + 1) + 1 for i in 1:Nx, j in 1:Ny]

    # The rows beneath the first one are neither read nor written: they hold non-finite right-hand sides
    # that must not reach the solved rows, and their solution keeps the sentinel value it is initialized with.
    sentinel = -1.0
    ϕ_correct = fill(sentinel, Nx, Ny, Nz)

    for i = 1:Nx, j = 1:Ny
        k₁ = first_rows[i, j]
        f[i, j, 1:k₁-1] .= NaN
        k₁ > Nz && continue
        M = Tridiagonal(a[k₁:Nz-1], b[k₁:Nz], c[k₁:Nz-1])
        ϕ_correct[i, j, k₁:Nz] .= M \ f[i, j, k₁:Nz]
    end

    # Convert to CuArray if needed.
    a, b, c, f, first_rows = ArrayType.([a, b, c, f, first_rows])

    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(1, 1, 1))
    btsolver = BatchedTridiagonalSolver(grid;
                                        lower_diagonal = a,
                                        diagonal = PartialColumnDiagonal(b, first_rows),
                                        upper_diagonal = c)

    ϕ = fill(sentinel, Nx, Ny, Nz) |> ArrayType

    solve!(ϕ, btsolver, f)

    return all(Array(ϕ) .≈ ϕ_correct)
end

@testset "Batched tridiagonal solvers" begin
    @info "Testing BatchedTridiagonalSolver..."

    for arch in archs
        @testset "Batched tridiagonal solver [$arch]" begin
            for Nx in [3, 8], Ny in [5, 16], Nz in [8, 11]
                @test can_solve_batched_tridiagonal_system_with_3D_RHS(arch, Nx, Ny, Nz)
                for tridiagonal_direction in (XDirection(), YDirection(), ZDirection())
                    @test can_solve_single_tridiagonal_system(arch, Nz; tridiagonal_direction)
                end
            end

            for Nx in [3, 8], Ny in [5, 16], Nz in [8, 11]
                for tridiagonal_direction in (XDirection(), YDirection(), ZDirection())
                    @test can_solve_batched_tridiagonal_system_with_3D_RHS(arch, Nx, Ny, Nz; tridiagonal_direction)
                end
            end

            for Nx in [3, 8], Ny in [5, 16], Nz in [8, 11]
                @test can_solve_partial_columns(arch, Nx, Ny, Nz)
            end
        end
    end
end
