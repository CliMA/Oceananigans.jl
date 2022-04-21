include("dependencies_for_runtests.jl")

using LinearAlgebra
using Oceananigans.Architectures: array_type

function can_solve_single_tridiagonal_system(arch, N)
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

    ϕ = reshape(zeros(N), (1, 1, N)) |> ArrayType

    grid = RectilinearGrid(arch, size=(1, 1, N), extent=(1, 1, 1))

    btsolver = BatchedTridiagonalSolver(grid;
                                        lower_diagonal = a,
                                        diagonal = b,
                                        upper_diagonal = c)

    solve!(ϕ, btsolver, f)

    return Array(ϕ[:]) ≈ ϕ_correct
end

function can_solve_single_tridiagonal_system_with_functions(arch, N)
    ArrayType = array_type(arch)

    grid = RectilinearGrid(arch, size=(1, 1, N), extent=(1, 1, 1))

    a = rand(N-1)
    c = rand(N-1)

    @inline b(i, j, k, grid) = 3 .+ cos(2π * grid.zᵃᵃᶜ[k])  # +3 to ensure diagonal dominance.
    @inline f(i, j, k, grid) = sin(2π * grid.zᵃᵃᶜ[k])

    bₐ = [b(1, 1, k, grid) for k in 1:N]
    fₐ = [f(1, 1, k, grid) for k in 1:N]

    # Solve the system with backslash on the CPU to avoid scalar operations on the GPU.
    M = Tridiagonal(a, bₐ, c)
    ϕ_correct = M \ fₐ

    # Convert to CuArray if needed.
    a, c = ArrayType.((a, c))

    ϕ = reshape(zeros(N), (1, 1, N)) |> ArrayType

    btsolver = BatchedTridiagonalSolver(grid;
                                        lower_diagonal = a,
                                        diagonal = b,
                                        upper_diagonal = c)

    solve!(ϕ, btsolver, f)

    return Array(ϕ[:]) ≈ ϕ_correct
end

function can_solve_batched_tridiagonal_system_with_3D_RHS(arch, Nx, Ny, Nz)
    ArrayType = array_type(arch)

    a = rand(Nz-1)
    b = 3 .+ rand(Nz) # +3 to ensure diagonal dominance.
    c = rand(Nz-1)
    f = rand(Nx, Ny, Nz)

    M = Tridiagonal(a, b, c)
    ϕ_correct = zeros(Nx, Ny, Nz)

    # Solve the systems with backslash on the CPU to avoid scalar operations on the GPU.
    for i = 1:Nx, j = 1:Ny
        ϕ_correct[i, j, :] .= M \ f[i, j, :]
    end

    # Convert to CuArray if needed.
    a, b, c, f = ArrayType.([a, b, c, f])

    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(1, 1, 1))
    btsolver = BatchedTridiagonalSolver(grid;
                                        lower_diagonal = a,
                                        diagonal = b,
                                        upper_diagonal = c)

    ϕ = zeros(Nx, Ny, Nz) |> ArrayType

    solve!(ϕ, btsolver, f)

    return Array(ϕ) ≈ ϕ_correct
end

function can_solve_batched_tridiagonal_system_with_3D_functions(arch, Nx, Ny, Nz)
    ArrayType = array_type(arch)

    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(1, 1, 1))

    a = rand(Nz-1)
    c = rand(Nz-1)

    @inline b(i, j, k, grid) = 3 + grid.xᶜᵃᵃ[i] * grid.yᵃᶜᵃ[j] * cos(2π * grid.zᵃᵃᶜ[k])
    @inline f(i, j, k, grid) = (grid.xᶜᵃᵃ[i] + grid.yᵃᶜᵃ[j]) * sin(2π * grid.zᵃᵃᶜ[k])

    ϕ_correct = zeros(Nx, Ny, Nz)

    # Solve the system with backslash on the CPU to avoid scalar operations on the GPU.
    for i = 1:Nx, j = 1:Ny
        bₐ = [b(i, j, k, grid) for k in 1:Nz]
        M = Tridiagonal(a, bₐ, c)

        fₐ = [f(i, j, k, grid) for k in 1:Nz]
        ϕ_correct[i, j, :] .= M \ fₐ
    end

    # Convert to CuArray if needed.
    a, c = ArrayType.([a, c])

    btsolver = BatchedTridiagonalSolver(grid;
                                        lower_diagonal = a,
                                        diagonal = b,
                                        upper_diagonal = c)

    ϕ = zeros(Nx, Ny, Nz) |> ArrayType
    solve!(ϕ, btsolver, f)

    return Array(ϕ) ≈ ϕ_correct
end

@testset "Batched tridiagonal solvers" begin
    @info "Testing BatchedTridiagonalSolver..."

    for arch in archs
        @testset "Batched tridiagonal solver [$arch]" begin
            for Nz in [8, 11, 18]
                @test can_solve_single_tridiagonal_system(arch, Nz)
                @test can_solve_single_tridiagonal_system_with_functions(arch, Nz)
            end

            for Nx in [3, 8], Ny in [5, 16], Nz in [8, 11]
                @test can_solve_batched_tridiagonal_system_with_3D_RHS(arch, Nx, Ny, Nz)
                @test can_solve_batched_tridiagonal_system_with_3D_functions(arch, Nx, Ny, Nz)
            end
        end
    end
end
