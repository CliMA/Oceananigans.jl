# The `ZDirection` sweep of the batched tridiagonal solver takes the values it has just stored from registers, runs
# its back-substitution four levels at a time and loads an array right-hand side one level ahead
# (`solve_batched_tridiagonal_system_z!`). It must leave the bits of the textbook loop below, which is the sweep as it
# was written before, in `ϕ` and in the scratch `t`: for array and function coefficients, a separate and an in-place
# right-hand side, mixed precision, complex right-hand sides, rows that are not diagonally dominant, decoupled rows
# holding NaN, and every remainder of the blocked back-substitution. Runs on the CPU:
#   julia --project=<env> test/test_batched_tridiagonal_sweep.jl
using Test
using Oceananigans
using Oceananigans.Grids: ZDirection
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!, get_coefficient, masked_multiply, float_eltype
import Oceananigans.Solvers: get_coefficient

# The sweep as it was: every value that was stored is loaded again, one level at a time.
function reference_sweep!(ϕ, a, b, c, f, t, grid, p, args, direction)
    Nx, Ny, Nz = size(grid)
    for i in 1:Nx, j in 1:Ny
        β  = get_coefficient(i, j, 1, grid, b, p, direction, args...)
        f₁ = get_coefficient(i, j, 1, grid, f, p, direction, args...)
        ϕ[i, j, 1] = f₁ / β

        for k = 2:Nz
            cᵏ⁻¹ = get_coefficient(i, j, k-1, grid, c, p, direction, args...)
            bᵏ   = get_coefficient(i, j, k,   grid, b, p, direction, args...)
            aᵏ⁻¹ = get_coefficient(i, j, k-1, grid, a, p, direction, args...)

            t[i, j, k] = cᵏ⁻¹ / β
            β = bᵏ - masked_multiply(aᵏ⁻¹, t[i, j, k])
            fᵏ = get_coefficient(i, j, k, grid, f, p, direction, args...)

            definitely_diagonally_dominant = abs(β) > 10 * eps(float_eltype(ϕ))
            ϕ★ = (fᵏ - masked_multiply(aᵏ⁻¹, ϕ[i, j, k-1])) / β
            definitely_diagonally_dominant && (ϕ[i, j, k] = ϕ★)
        end

        for k = Nz-1:-1:1
            ϕ[i, j, k] -= masked_multiply(t[i, j, k+1], ϕ[i, j, k+1])
        end
    end
    return nothing
end

unsigned_of(::Type{Float32}) = UInt32
unsigned_of(::Type{Float64}) = UInt64
bits(a::AbstractArray{T}) where T <: AbstractFloat = reinterpret(unsigned_of(T), a)
bits(a::AbstractArray{Complex{T}}) where T = reinterpret(unsigned_of(T), a)

# Function coefficients (as the vertically implicit closures pass them): rows that depend on the arguments and on k,
# with a zero off-diagonal across a "wall" that decouples part of the column.
struct LowerRow end
struct DiagonalRow end
struct UpperRow end
@inline row_strength(i, j, k, κ, wall) = ifelse(k == wall, zero(eltype(κ)), @inbounds κ[i, j, k])
@inline get_coefficient(i, j, k, grid, ::LowerRow, p, ::ZDirection, κ, Δt, wall) = - Δt * row_strength(i, j, k + 1, κ, wall)
@inline get_coefficient(i, j, k, grid, ::UpperRow, p, ::ZDirection, κ, Δt, wall) = - Δt * row_strength(i, j, k + 1, κ, wall)
@inline get_coefficient(i, j, k, grid, ::DiagonalRow, p, ::ZDirection, κ, Δt, wall) =
    1 + Δt * (row_strength(i, j, k, κ, wall) + row_strength(i, j, k + 1, κ, wall))

function compare_sweeps(ϕ₀, a, b, c, f, T, grid, args...; in_place = false)
    solver = BatchedTridiagonalSolver(grid; lower_diagonal = a, diagonal = b, upper_diagonal = c,
                                      scratch = zeros(T, size(grid)...))
    reference_t = fill!(similar(solver.t), 0)

    ϕ = copy(ϕ₀)
    rhs = in_place ? ϕ : f
    solve!(ϕ, solver, rhs, args...)

    ϕʳ = copy(ϕ₀)
    rhsʳ = in_place ? ϕʳ : f
    reference_sweep!(ϕʳ, a, b, c, rhsʳ, reference_t, grid, nothing, args, ZDirection())

    return bits(ϕ) == bits(ϕʳ) && bits(solver.t) == bits(reference_t)
end

@testset "ZDirection sweep against the textbook loop" begin
    for Nz in (1, 2, 3, 4, 5, 6, 8, 9, 12, 17), FT in (Float32, Float64)
        Nx, Ny = 3, 2
        grid = RectilinearGrid(CPU(), FT; size = (Nx, Ny, Nz), extent = (1, 1, 1))
        N = (Nx, Ny, Nz)

        # three-dimensional array coefficients, separate right-hand side, old ϕ present (the fallback keeps it)
        a = rand(FT, N...); c = rand(FT, N...); b = 3 .+ rand(FT, N...)
        f = randn(FT, N...); ϕ₀ = randn(FT, N...)
        @test compare_sweeps(ϕ₀, a, b, c, f, FT, grid)
        @test compare_sweeps(ϕ₀, a, b, c, f, FT, grid; in_place = true)

        # one-dimensional coefficients and right-hand side (the right-hand side keeps its place in the sweep)
        a₁ = rand(FT, Nz); c₁ = rand(FT, Nz); b₁ = 3 .+ rand(FT, Nz); f₁ = randn(FT, Nz)
        @test compare_sweeps(ϕ₀, a₁, b₁, c₁, f₁, FT, grid)
        @test compare_sweeps(ϕ₀, a₁, b₁, c₁, f, FT, grid)

        # rows that are not diagonally dominant: β is exactly zero at some levels, the fallback keeps ϕ
        if Nz ≥ 3
            bz = copy(b); az = copy(a); cz = copy(c)
            for i in 1:Nx, j in 1:Ny, k in 2:2:Nz
                # β = b[k] - a[k-1] c[k-1] / β[k-1] = 0 with exact small integers
                bz[i, j, k-1] = 2; cz[i, j, k-1] = 4; az[i, j, k-1] = 1; bz[i, j, k] = 2
                k == 2 || (az[i, j, k-2] = 0)
            end
            @test compare_sweeps(ϕ₀, az, bz, cz, f, FT, grid)
            @test compare_sweeps(ϕ₀, az, bz, cz, f, FT, grid; in_place = true)
        end

        # mixed precision: Float64 rows from function coefficients on Float32 or Float64 fields, a decoupled lower part
        # of the column holding NaN (the masked products keep it there), in place and with a separate right-hand side
        κ = rand(FT, Nx, Ny, Nz + 1)
        for wall in (0, max(1, Nz ÷ 2))
            ϕₙ = copy(ϕ₀)
            wall > 0 && (ϕₙ[:, :, 1:wall] .= NaN)
            fₙ = copy(ϕₙ)
            @test compare_sweeps(ϕₙ, LowerRow(), DiagonalRow(), UpperRow(), fₙ, FT, grid, κ, 0.37, wall + 1)
            @test compare_sweeps(ϕₙ, LowerRow(), DiagonalRow(), UpperRow(), fₙ, FT, grid, κ, 0.37, wall + 1; in_place = true)
            @test compare_sweeps(ϕₙ, LowerRow(), DiagonalRow(), UpperRow(), fₙ, Float32, grid, κ, 0.37f0, wall + 1)
        end

        # complex right-hand side and solution with real rows (the Fourier-tridiagonal Poisson solvers)
        fᶜ = randn(Complex{FT}, N...); ϕᶜ = randn(Complex{FT}, N...)
        @test compare_sweeps(ϕᶜ, a₁, b₁, c₁, fᶜ, FT, grid)
        @test compare_sweeps(ϕᶜ, a, b, c, fᶜ, FT, grid; in_place = true)
    end
end
