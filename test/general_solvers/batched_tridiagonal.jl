include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using LinearAlgebra
using Random
using Oceananigans.Architectures: array_type
using Oceananigans.Grids: XDirection, YDirection, ZDirection
using Oceananigans.Solvers: system_size

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

# Garbage rows have non-finite diagonals and right-hand sides but vanishing couplings, like inactive
# cells across an immersed boundary; the remaining rows must still solve their own, smaller system.
function can_isolate_decoupled_rows(arch, FT, Nx, Ny, Nz; garbage_end = :bottom)
    ArrayType = array_type(arch)

    a = rand(FT, Nx, Ny, Nz-1)
    b = 3 .+ rand(FT, Nx, Ny, Nz) # +3 to ensure diagonal dominance.
    c = rand(FT, Nx, Ny, Nz-1)
    f = rand(FT, Nx, Ny, Nz)

    expected_solution = zeros(FT, Nx, Ny, Nz)
    remaining = falses(Nx, Ny, Nz)

    for i = 1:Nx, j = 1:Ny
        # Every column has a different number of garbage rows, from none to all of them
        m = mod(i - 1 + Nx * (j - 1), Nz + 1)
        garbage_rows   = garbage_end == :bottom ? (1:m) : (Nz-m+1:Nz)
        remaining_rows = garbage_end == :bottom ? (m+1:Nz) : (1:Nz-m)

        for k in garbage_rows
            b[i, j, k] = NaN
            f[i, j, k] = NaN
            k > 1  && (a[i, j, k-1] = 0; c[i, j, k-1] = 0) # couplings between rows k-1 and k
            k < Nz && (a[i, j, k]   = 0; c[i, j, k]   = 0) # couplings between rows k and k+1
        end

        isempty(remaining_rows) && continue
        M = Tridiagonal(a[i, j, remaining_rows[1:end-1]], b[i, j, remaining_rows], c[i, j, remaining_rows[1:end-1]])
        expected_solution[i, j, remaining_rows] .= M \ f[i, j, remaining_rows]
        remaining[i, j, remaining_rows] .= true
    end

    # Convert to CuArray if needed.
    a, b, c, f = ArrayType.([a, b, c, f])

    grid = RectilinearGrid(arch, FT, size=(Nx, Ny, Nz), extent=(1, 1, 1))
    btsolver = BatchedTridiagonalSolver(grid;
                                        lower_diagonal = a,
                                        diagonal = b,
                                        upper_diagonal = c)

    ϕ = zeros(FT, Nx, Ny, Nz) |> ArrayType

    solve!(ϕ, btsolver, f)
    ϕ = Array(ϕ)

    return all(isfinite, ϕ[remaining]) && ϕ[remaining] ≈ expected_solution[remaining]
end

# `solve_factorized!` with the factors of `factorize!` must reproduce `solve!` bitwise, for 1D and 3D
# coefficients in every direction, and for a second right hand side. `isequal` treats NaNs as equal,
# which matters for the decoupled-row (immersed) case below.
function factorized_solve_is_bitwise_identical(arch, FT, Nx, Ny, Nz; tridiagonal_direction = ZDirection(), coefficient_dims = 3)
    ArrayType = array_type(arch)
    rng = MersenneTwister(7)

    grid = RectilinearGrid(arch, FT; size = (Nx, Ny, Nz), extent = (1, 1, 1))
    N = size(grid, Oceananigans.Solvers.sweep_dimension(tridiagonal_direction))
    sz = system_size(grid, tridiagonal_direction, N)
    coefficient_size = coefficient_dims == 1 ? (N,) : sz

    a = randn(rng, FT, coefficient_size...)
    b = FT(3) .+ rand(rng, FT, coefficient_size...)
    c = randn(rng, FT, coefficient_size...)
    f = randn(rng, FT, sz...)
    g = randn(rng, FT, sz...)
    a, b, c, f, g = ArrayType.([a, b, c, f, g])

    solver = BatchedTridiagonalSolver(grid; lower_diagonal = a, diagonal = b, upper_diagonal = c,
                                      tridiagonal_direction, store_factors = true)

    ϕ = ArrayType(zeros(FT, sz...))
    ψ = ArrayType(zeros(FT, sz...))
    solve!(ϕ, solver, f)
    solve!(ψ, solver, g)

    factorize!(solver)
    ϕ_factorized = ArrayType(zeros(FT, sz...))
    ψ_factorized = ArrayType(zeros(FT, sz...))
    solve_factorized!(ϕ_factorized, solver, f)
    solve_factorized!(ψ_factorized, solver, g)

    return isequal(Array(ϕ_factorized), Array(ϕ)) && isequal(Array(ψ_factorized), Array(ψ))
end

# The masking of the `ZDirection` sweep must survive the factorization: with NaN garbage rows decoupled
# from the rest of the column, the factored solve must be bitwise identical to the direct one, NaNs included.
function factorized_solve_isolates_decoupled_rows(arch, FT, Nx, Ny, Nz)
    ArrayType = array_type(arch)

    a = rand(FT, Nx, Ny, Nz-1)
    b = 3 .+ rand(FT, Nx, Ny, Nz) # +3 to ensure diagonal dominance.
    c = rand(FT, Nx, Ny, Nz-1)
    f = rand(FT, Nx, Ny, Nz)

    for i = 1:Nx, j = 1:Ny
        m = mod(i - 1 + Nx * (j - 1), Nz + 1)
        for k in 1:m
            b[i, j, k] = NaN
            f[i, j, k] = NaN
            k > 1  && (a[i, j, k-1] = 0; c[i, j, k-1] = 0)
            k < Nz && (a[i, j, k]   = 0; c[i, j, k]   = 0)
        end
    end

    a, b, c, f = ArrayType.([a, b, c, f])

    grid = RectilinearGrid(arch, FT, size=(Nx, Ny, Nz), extent=(1, 1, 1))
    solver = BatchedTridiagonalSolver(grid; lower_diagonal = a, diagonal = b, upper_diagonal = c, store_factors = true)

    ϕ = ArrayType(zeros(FT, Nx, Ny, Nz))
    solve!(ϕ, solver, f)

    factorize!(solver)
    ϕ_factorized = ArrayType(zeros(FT, Nx, Ny, Nz))
    solve_factorized!(ϕ_factorized, solver, f)

    return any(isnan, Array(ϕ)) && isequal(Array(ϕ_factorized), Array(ϕ))
end

@testset "Batched tridiagonal solvers" begin
    @testset "Batched tridiagonal solver [$arch]" for arch in archs
        for Nx in [3, 8], Ny in [5, 16], Nz in [8, 11]
            @test can_solve_batched_tridiagonal_system_with_3D_RHS(arch, Nx, Ny, Nz)
        end

        for Nz in [8, 11], tridiagonal_direction in (XDirection(), YDirection(), ZDirection())
            @test can_solve_single_tridiagonal_system(arch, Nz; tridiagonal_direction)
        end

        for Nx in [3, 8], Ny in [5, 16], Nz in [8, 11]
            for tridiagonal_direction in (XDirection(), YDirection(), ZDirection())
                @test can_solve_batched_tridiagonal_system_with_3D_RHS(arch, Nx, Ny, Nz; tridiagonal_direction)
            end
        end

        for FT in float_types, Nx in [3, 8], Ny in [5, 16], Nz in [8, 11], garbage_end in (:bottom, :top)
            @test can_isolate_decoupled_rows(arch, FT, Nx, Ny, Nz; garbage_end)
        end

        for FT in float_types, coefficient_dims in (1, 3), tridiagonal_direction in (XDirection(), YDirection(), ZDirection())
            @test factorized_solve_is_bitwise_identical(arch, FT, 5, 4, 9; tridiagonal_direction, coefficient_dims)
        end

        for FT in float_types, Nz in [8, 11]
            @test factorized_solve_isolates_decoupled_rows(arch, FT, 3, 5, Nz)
        end

        grid = RectilinearGrid(arch, size = (2, 2, 4), extent = (1, 1, 1))
        solver = BatchedTridiagonalSolver(grid; lower_diagonal = zeros(3), diagonal = ones(4), upper_diagonal = zeros(3))
        @test_throws ArgumentError factorize!(solver)
        @test_throws ArgumentError solve_factorized!(zeros(2, 2, 4), solver, ones(2, 2, 4))
    end
end
