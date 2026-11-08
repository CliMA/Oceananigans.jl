include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using LinearAlgebra
using Random
using Oceananigans.Architectures: array_type
using Oceananigans.Grids: XDirection, YDirection, ZDirection
using Oceananigans.Solvers: get_coefficient, sweep_dimension, system_size

#####
##### Callable coefficient tags: the coefficient of row n in column (i, j) is computed from the
##### solver `parameters` and the trailing `args...` of `solve!` / `factorize!`.
#####

struct ScaledBand{B}
    band :: B
end

struct ShiftedDiagonal{B}
    band :: B
end

@inline Oceananigans.Solvers.get_coefficient(i, j, k, grid, c::ScaledBand, parameters, direction, shift, scale) =
    scale * get_coefficient(i, j, k, grid, c.band, parameters, direction)

@inline Oceananigans.Solvers.get_coefficient(i, j, k, grid, c::ShiftedDiagonal, parameters, direction, shift, scale) =
    parameters.dominance * shift + get_coefficient(i, j, k, grid, c.band, parameters, direction)

#####
##### Dense reference: the pentadiagonal matrix of every system, built with the same accessor the
##### solver uses (band-by-row convention), and solved with LinearAlgebra on the CPU.
#####

function dense_pentadiagonal(grid, bands, N, column, direction, parameters, args)
    e, a, b, c, d = bands
    A = zeros(eltype(grid), N, N)
    for n in 1:N
        i, j, k = Oceananigans.Solvers.column_indices(direction, n, column...)
        A[n, n] = get_coefficient(i, j, k, grid, b, parameters, direction, args...)
        n > 1     && (A[n, n-1] = get_coefficient(i, j, k, grid, a, parameters, direction, args...))
        n > 2     && (A[n, n-2] = get_coefficient(i, j, k, grid, e, parameters, direction, args...))
        n < N     && (A[n, n+1] = get_coefficient(i, j, k, grid, c, parameters, direction, args...))
        n < N - 1 && (A[n, n+2] = get_coefficient(i, j, k, grid, d, parameters, direction, args...))
    end
    return A
end

function dense_pentadiagonal_solution(grid, bands, f, N, direction, parameters, args)
    FT = eltype(grid)
    sz = system_size(grid, direction, N)
    ϕ = zeros(FT, sz...)
    dim = sweep_dimension(direction)
    column_dims = Tuple(m for m in 1:3 if m != dim)
    for column in Iterators.product((1:sz[m] for m in column_dims)...)
        A = dense_pentadiagonal(grid, bands, N, column, direction, parameters, args)
        rhs = [get_coefficient(Oceananigans.Solvers.column_indices(direction, n, column...)..., grid, f, parameters, direction, args...) for n in 1:N]
        x = A \ rhs
        for n in 1:N
            ϕ[Oceananigans.Solvers.column_indices(direction, n, column...)...] = x[n]
        end
    end
    return ϕ
end

# Random bands whose diagonal dominates the sum of the off-diagonals, so that LU without pivoting is stable.
function random_bands(rng, FT, sz; dominance = 6)
    e = randn(rng, FT, sz...)
    a = randn(rng, FT, sz...)
    b = FT(dominance) .+ rand(rng, FT, sz...)
    c = randn(rng, FT, sz...)
    d = randn(rng, FT, sz...)
    return (e, a, b, c, d)
end

relative_error(ϕ, ϕ_correct) = maximum(abs, Array(ϕ) .- ϕ_correct) / maximum(abs, ϕ_correct)
tolerance(FT) = FT == Float64 ? 1e-11 : 1e-4

# `coefficient_form` selects how the bands are supplied: `:array3D`, `:array1D`, or `:callable`.
# The reference is evaluated on CPU copies of the same coefficient objects.
function can_solve_batched_pentadiagonal_system(arch, FT, Nx, Ny, Nz; N = nothing, direction = ZDirection(), coefficient_form = :array3D, seed = 1)
    ArrayType = array_type(arch)
    rng = MersenneTwister(seed)

    grid = RectilinearGrid(arch, FT; size = (Nx, Ny, Nz), extent = (1, 1, 1))
    cpu_grid = RectilinearGrid(CPU(), FT; size = (Nx, Ny, Nz), extent = (1, 1, 1))
    N = isnothing(N) ? size(grid, sweep_dimension(direction)) : N
    sz = system_size(grid, direction, N)

    parameters = nothing
    args = ()

    if coefficient_form == :array1D
        cpu_bands = random_bands(rng, FT, (N,))
    else
        cpu_bands = random_bands(rng, FT, sz; dominance = coefficient_form == :callable ? 0 : 6)
    end

    if coefficient_form == :callable
        # Diagonal dominance enters through `parameters` and `args`, so a wrongly routed argument fails the test.
        shift, scale = FT(1), FT(0.5)
        parameters = (; dominance = FT(6))
        args = (shift, scale)
        e, a, b, c, d = cpu_bands
        cpu_bands = (ScaledBand(e), ScaledBand(a), ShiftedDiagonal(b), ScaledBand(c), ScaledBand(d))
    end

    cpu_f = randn(rng, FT, sz...)
    ϕ_correct = dense_pentadiagonal_solution(cpu_grid, cpu_bands, cpu_f, N, direction, parameters, args)

    to_arch(x::AbstractArray) = ArrayType(x)
    to_arch(x::ScaledBand) = ScaledBand(ArrayType(x.band))
    to_arch(x::ShiftedDiagonal) = ShiftedDiagonal(ArrayType(x.band))
    e, a, b, c, d = map(to_arch, cpu_bands)
    f = ArrayType(cpu_f)

    solver = BatchedPentadiagonalSolver(grid; N, direction, parameters,
                                        second_lower_diagonal = e,
                                        lower_diagonal = a,
                                        diagonal = b,
                                        upper_diagonal = c,
                                        second_upper_diagonal = d,
                                        store_factors = true)

    ϕ = ArrayType(zeros(FT, sz...))
    solve!(ϕ, solver, f, args...)
    on_the_fly_is_correct = relative_error(ϕ, ϕ_correct) < tolerance(FT)

    factorize!(solver, args...)
    ϕ_factorized = ArrayType(zeros(FT, sz...))
    solve_factorized!(ϕ_factorized, solver, f, args...)
    factorized_is_bitwise_identical = Array(ϕ_factorized) == Array(ϕ)

    # A second right hand side reuses the stored factors.
    cpu_g = randn(rng, FT, sz...)
    ψ_correct = dense_pentadiagonal_solution(cpu_grid, cpu_bands, cpu_g, N, direction, parameters, args)
    ψ = ArrayType(zeros(FT, sz...))
    solve_factorized!(ψ, solver, ArrayType(cpu_g), args...)
    second_rhs_is_correct = relative_error(ψ, ψ_correct) < tolerance(FT)

    return on_the_fly_is_correct && factorized_is_bitwise_identical && second_rhs_is_correct
end

# Two unknowns per cell, (w, θ) on a staggered column, interleaved as (w¹, θ¹, w², θ², …) into one system of
# N = 2Nz unknowns per column. With the couplings of an acoustic system — w at face k to θ in cells k-1 and k
# (pressure gradient) and to w at faces k±1 (implicit vertical advection); θ in cell k to w at faces k and k+1
# (divergence) and to θ in cells k±1 — the interleaved matrix is pentadiagonal with all five bands populated.
function can_solve_interleaved_coupled_system(arch, FT, Nx, Ny, Nz)
    ArrayType = array_type(arch)
    rng = MersenneTwister(3)
    N = 2Nz

    grid = RectilinearGrid(arch, FT; size = (Nx, Ny, Nz), extent = (1, 1, 1))

    # Dense N × N matrices, one per column; rows 2k-1 and 2k hold the w and θ equations of cell k
    A = zeros(FT, Nx, Ny, N, N)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        w, θ = 2k-1, 2k
        A[i, j, w, w] = FT(6) + rand(rng, FT)
        A[i, j, θ, θ] = FT(6) + rand(rng, FT)
        A[i, j, w, θ] = randn(rng, FT)                 # w ← θ of the same cell
        A[i, j, θ, w] = randn(rng, FT)                 # θ ← w of the same cell
        if k > 1
            A[i, j, w, w-2] = randn(rng, FT)           # w ← w below
            A[i, j, w, θ-2] = randn(rng, FT)           # w ← θ below
            A[i, j, θ, θ-2] = randn(rng, FT)           # θ ← θ below
        end
        if k < Nz
            A[i, j, w, w+2] = randn(rng, FT)           # w ← w above
            A[i, j, θ, w+2] = randn(rng, FT)           # θ ← w above
            A[i, j, θ, θ+2] = randn(rng, FT)           # θ ← θ above
        end
    end

    # Extract the five bands, by row
    e = zeros(FT, Nx, Ny, N); a = zeros(FT, Nx, Ny, N); b = zeros(FT, Nx, Ny, N)
    c = zeros(FT, Nx, Ny, N); d = zeros(FT, Nx, Ny, N)
    for i in 1:Nx, j in 1:Ny, n in 1:N
        b[i, j, n] = A[i, j, n, n]
        n > 1     && (a[i, j, n] = A[i, j, n, n-1])
        n > 2     && (e[i, j, n] = A[i, j, n, n-2])
        n < N     && (c[i, j, n] = A[i, j, n, n+1])
        n < N - 1 && (d[i, j, n] = A[i, j, n, n+2])
    end

    # The band representation must be exact
    for i in 1:Nx, j in 1:Ny, n in 1:N, m in 1:N
        abs(n - m) > 2 && A[i, j, n, m] != 0 && return false
    end

    cpu_f = randn(rng, FT, Nx, Ny, N)
    ϕ_correct = zeros(FT, Nx, Ny, N)
    for i in 1:Nx, j in 1:Ny
        ϕ_correct[i, j, :] .= A[i, j, :, :] \ cpu_f[i, j, :]
    end

    e, a, b, c, d, f = ArrayType.([e, a, b, c, d, cpu_f])
    solver = BatchedPentadiagonalSolver(grid; N,
                                        second_lower_diagonal = e,
                                        lower_diagonal = a,
                                        diagonal = b,
                                        upper_diagonal = c,
                                        second_upper_diagonal = d,
                                        store_factors = true)

    ϕ = ArrayType(zeros(FT, Nx, Ny, N))
    solve!(ϕ, solver, f)
    on_the_fly_is_correct = relative_error(ϕ, ϕ_correct) < tolerance(FT)

    factorize!(solver)
    ϕ_factorized = ArrayType(zeros(FT, Nx, Ny, N))
    solve_factorized!(ϕ_factorized, solver, f)
    factorized_is_bitwise_identical = Array(ϕ_factorized) == Array(ϕ)

    return on_the_fly_is_correct && factorized_is_bitwise_identical
end

# A pentadiagonal system with vanishing second off-diagonals is tridiagonal; the pentadiagonal solver must
# then agree with `BatchedTridiagonalSolver` to rounding (the index conventions of the lower diagonal differ).
function pentadiagonal_reduces_to_tridiagonal(arch, FT, Nx, Ny, Nz)
    ArrayType = array_type(arch)
    rng = MersenneTwister(5)

    grid = RectilinearGrid(arch, FT; size = (Nx, Ny, Nz), extent = (1, 1, 1))
    lower = randn(rng, FT, Nx, Ny, Nz)   # tridiagonal convention: lower[k] multiplies ϕ[k] in row k+1
    b = FT(4) .+ rand(rng, FT, Nx, Ny, Nz)
    c = randn(rng, FT, Nx, Ny, Nz)
    f = randn(rng, FT, Nx, Ny, Nz)

    a = zeros(FT, Nx, Ny, Nz)            # pentadiagonal convention: a[k] multiplies ϕ[k-1] in row k
    a[:, :, 2:Nz] .= lower[:, :, 1:Nz-1]
    zero_band = zeros(FT, Nx, Ny, Nz)

    lower, a, b, c, f, zero_band = ArrayType.([lower, a, b, c, f, zero_band])

    tridiagonal_solver = BatchedTridiagonalSolver(grid; lower_diagonal = lower, diagonal = b, upper_diagonal = c)
    pentadiagonal_solver = BatchedPentadiagonalSolver(grid; second_lower_diagonal = zero_band, lower_diagonal = a,
                                                      diagonal = b, upper_diagonal = c, second_upper_diagonal = zero_band)

    ϕ₃ = ArrayType(zeros(FT, Nx, Ny, Nz))
    ϕ₅ = ArrayType(zeros(FT, Nx, Ny, Nz))
    solve!(ϕ₃, tridiagonal_solver, f)
    solve!(ϕ₅, pentadiagonal_solver, f)

    return relative_error(ϕ₅, Array(ϕ₃)) < tolerance(FT)
end

@testset "Batched pentadiagonal solvers" begin
    @testset "Batched pentadiagonal solver [$arch]" for arch in archs
        for FT in float_types, coefficient_form in (:array3D, :array1D, :callable)
            for direction in (XDirection(), YDirection(), ZDirection())
                @test can_solve_batched_pentadiagonal_system(arch, FT, 5, 4, 9; direction, coefficient_form)
            end

            for Nx in [3, 8], Ny in [5, 16], Nz in [8, 11]
                @test can_solve_batched_pentadiagonal_system(arch, FT, Nx, Ny, Nz; coefficient_form)
            end

            # N differs from the grid size in the solve direction
            @test can_solve_batched_pentadiagonal_system(arch, FT, 3, 4, 6; N = 12, coefficient_form)
            @test can_solve_batched_pentadiagonal_system(arch, FT, 6, 4, 3; N = 12, direction = XDirection(), coefficient_form)
        end

        for FT in float_types, Nz in [4, 11]
            @test can_solve_interleaved_coupled_system(arch, FT, 3, 5, Nz)
            @test pentadiagonal_reduces_to_tridiagonal(arch, FT, 3, 5, Nz)
        end

        grid = RectilinearGrid(arch, size = (2, 2, 4), extent = (1, 1, 1))
        bands = (; second_lower_diagonal = zeros(4), lower_diagonal = zeros(4), diagonal = ones(4),
                   upper_diagonal = zeros(4), second_upper_diagonal = zeros(4))
        @test_throws ArgumentError BatchedPentadiagonalSolver(grid; N = 2, bands...)
        solver = BatchedPentadiagonalSolver(grid; bands...)
        @test_throws ArgumentError factorize!(solver)
        @test_throws ArgumentError solve_factorized!(zeros(2, 2, 4), solver, ones(2, 2, 4))
    end
end
