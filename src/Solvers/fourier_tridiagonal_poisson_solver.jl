using Oceananigans.Operators: Δxᶜᵃᵃ, Δxᶠᵃᵃ, Δyᵃᶜᵃ, Δyᵃᶠᵃ, Δzᵃᵃᶜ, Δzᵃᵃᶠ
using Oceananigans.Grids: XYRegRectilinearGrid, XZRegRectilinearGrid, YZRegRectilinearGrid
import Oceananigans.Architectures: architecture

struct FourierTridiagonalPoissonSolver{G, B, R, S, β, T}
                          grid :: G
    batched_tridiagonal_solver :: B
                   source_term :: R
                       storage :: S
                        buffer :: β
                    transforms :: T
end

architecture(solver::FourierTridiagonalPoissonSolver) = architecture(solver.grid)

@kernel function compute_main_diagonals!(D, grid, λ1, λ2, ::XDirection)
    m, n = @index(Global, NTuple)
    N = getindex(size(grid), 1)

    # Using a homogeneous Neumann (zero Gradient) boundary condition:
    D[1, m, n] = -1 / Δxᶠᵃᵃ(2, m, n, grid) - Δxᶜᵃᵃ(1, m, n, grid) * (λ1[m] + λ2[n])
    @unroll for q in 2:N-1
        D[q, m, n] = - (1 / Δxᶠᵃᵃ(q+1, m, n, grid) + 1 / Δxᶠᵃᵃ(q, m, n, grid)) - Δxᶜᵃᵃ(q, m, n, grid) * (λ1[m] + λ2[n])
    end
    D[N, m, n] = -1 / Δxᶠᵃᵃ(N, m, n, grid) - Δxᶜᵃᵃ(N, m, n, grid) * (λ1[m] + λ2[n])
end 

@kernel function compute_main_diagonals!(D, grid, λ1, λ2, ::YDirection)
    m, n = @index(Global, NTuple)
    N = getindex(size(grid), 2)

    # Using a homogeneous Neumann (zero Gradient) boundary condition:
    D[m, 1, n] = -1 / Δyᵃᶠᵃ(m, 2, n, grid) - Δyᵃᶜᵃ(m, 1, n, grid) * (λ1[m] + λ2[n])
    @unroll for q in 2:N-1
        D[m, q, n] = - (1 / Δyᵃᶠᵃ(m, q+1, n, grid) + 1 / Δyᵃᶠᵃ(m, q, n, grid)) - Δyᵃᶜᵃ(m, q, n, grid) * (λ1[m] + λ2[n])
    end
    D[m, N, n] = -1 / Δyᵃᶠᵃ(m, N, n, grid) - Δyᵃᶜᵃ(m, N, n, grid) * (λ1[m] + λ2[n])
end 

@kernel function compute_main_diagonals!(D, grid, λ1, λ2, ::ZDirection)
    m, n = @index(Global, NTuple)
    N = getindex(size(grid), 3)

    # Using a homogeneous Neumann (zero Gradient) boundary condition:
    D[m, n, 1] = -1 / Δzᵃᵃᶠ(m, n, 2, grid) - Δzᵃᵃᶜ(m, n, 1, grid) * (λ1[m] + λ2[n])
    @unroll for q in 2:N-1
        D[m, n, q] = - (1 / Δzᵃᵃᶠ(m, n, q+1, grid) + 1 / Δzᵃᵃᶠ(m, n, q, grid)) - Δzᵃᵃᶜ(m, n, q, grid) * (λ1[m] + λ2[n])
    end
    D[m, n, N] = -1 / Δzᵃᵃᶠ(m, n, N, grid) - Δzᵃᵃᶜ(m, n, N, grid) * (λ1[m] + λ2[n])
end 


irregular_axis(::YZRegRectilinearGrid) = 1
irregular_axis(::XZRegRectilinearGrid) = 2
irregular_axis(::XYRegRectilinearGrid) = 3

irregular_direction(::YZRegRectilinearGrid) = XDirection()
irregular_direction(::XZRegRectilinearGrid) = YDirection()
irregular_direction(::XYRegRectilinearGrid) = ZDirection()

Δξᶠ(i, grid::YZRegRectilinearGrid) = Δxᶠᵃᵃ(i, 1, 1, grid)
Δξᶠ(j, grid::XZRegRectilinearGrid) = Δyᵃᶠᵃ(1, j, 1, grid)
Δξᶠ(k, grid::XYRegRectilinearGrid) = Δzᵃᵃᶠ(1, 1, k, grid)

extent(grid) = (grid.Lx, grid.Ly, grid.Lz)

function FourierTridiagonalPoissonSolver(grid, planner_flag=FFTW.PATIENT)
    irreg_axis = irregular_axis(grid)

    regular_top1, regular_top2 = Tuple( el for (i, el) in enumerate(topology(grid)) if i ≠ irreg_axis)
    regular_siz1, regular_siz2 = Tuple( el for (i, el) in enumerate(size(grid))     if i ≠ irreg_axis)
    regular_ext1, regular_ext2 = Tuple( el for (i, el) in enumerate(extent(grid))   if i ≠ irreg_axis)

    getindex(topology(grid), irreg_axis) != Bounded && error("`FourierTridiagonalPoissonSolver` can only be used when the irregular direction's topology is `Bounded`.")

    # Compute discrete Poisson eigenvalues
    λ1 = poisson_eigenvalues(regular_siz1, regular_ext1, 1, regular_top1())
    λ2 = poisson_eigenvalues(regular_siz2, regular_ext2, 2, regular_top2())

    arch = architecture(grid)
    λ1 = arch_array(arch, λ1)
    λ2 = arch_array(arch, λ2)

    # Plan required transforms for x and y
    sol_storage = arch_array(arch, zeros(complex(eltype(grid)), size(grid)...))
    transforms = plan_transforms(grid, sol_storage, planner_flag)

    # Lower and upper diagonals are the same
    lower_diagonal = CUDA.@allowscalar [ 1 / Δξᶠ(q, grid) for q in 2:size(grid)[irreg_axis] ]
    lower_diagonal = arch_array(arch, lower_diagonal)
    upper_diagonal = lower_diagonal

    # Compute diagonal coefficients for each grid point
    diagonal = arch_array(arch, zeros(size(grid)...))
    launch_config = if grid isa YZRegRectilinearGrid
                        :yz
                    elseif grid isa XZRegRectilinearGrid
                        :xz
                    elseif grid isa XYRegRectilinearGrid
                        :xy
                    end
    launch!(arch, grid, launch_config, compute_main_diagonals!, diagonal, grid, λ1, λ2, irregular_direction(grid))
    
    # Set up batched tridiagonal solver
    btsolver = BatchedTridiagonalSolver(grid; lower_diagonal, diagonal, upper_diagonal)

    # Need buffer for index permutations and transposes.
    buffer_needed = arch isa GPU && Bounded in (regular_top1, regular_top2)
    buffer = buffer_needed ? similar(sol_storage) : nothing

    # Storage space for right hand side of Poisson equation
    rhs = arch_array(arch, zeros(complex(eltype(grid)), size(grid)...))

    return FourierTridiagonalPoissonSolver(grid, btsolver, rhs, sol_storage, buffer, transforms)
end

function solve!(x, solver::FourierTridiagonalPoissonSolver, b=nothing)
    !isnothing(b) && set_source_term!(solver, b) # otherwise, assume source term is set correctly

    arch = architecture(solver)
    ϕ = solver.storage

    # Apply forward transforms in order
    [transform!(solver.source_term, solver.buffer) for transform! in solver.transforms.forward]

    # Solve tridiagonal system of linear equations in z at every column.
    solve!(ϕ, solver.batched_tridiagonal_solver, solver.source_term)

    # Apply backward transforms in order
    [transform!(ϕ, solver.buffer) for transform! in solver.transforms.backward]

    ϕ .= real.(ϕ)

    # Set the volume mean of the solution to be zero.
    # Solutions to Poisson's equation are only unique up to a constant (the global mean
    # of the solution), so we need to pick a constant. We choose the constant to be zero
    # so that the solution has zero-mean.
    ϕ .= ϕ .- mean(ϕ)

    launch!(arch, solver.grid, :xyz, copy_real_component!, x, ϕ, indices(x))
    
    return nothing
end

"""
    set_source_term!(solver, source_term)

Sets the source term in the discrete Poisson equation `solver` to `source_term` by
multiplying it by the vertical grid spacing at ``z`` cell centers.
"""
function set_source_term!(solver::FourierTridiagonalPoissonSolver, source_term)
    grid = solver.grid
    arch = architecture(solver)
    solver.source_term .= source_term

    launch!(arch, grid, :xyz, multiply_by_irregular_spacing!, solver.source_term, grid)

    return nothing
end


@kernel function multiply_by_irregular_spacing!(a, grid::YZRegRectilinearGrid)
    i, j, k = @index(Global, NTuple)
    a[i, j, k] *= Δxᶜᵃᵃ(i, j, k, grid)
end

@kernel function multiply_by_irregular_spacing!(a, grid::XZRegRectilinearGrid)
    i, j, k = @index(Global, NTuple)
    a[i, j, k] *= Δyᵃᶜᵃ(i, j, k, grid)
end

@kernel function multiply_by_irregular_spacing!(a, grid::XYRegRectilinearGrid)
    i, j, k = @index(Global, NTuple)
    a[i, j, k] *= Δzᵃᵃᶜ(i, j, k, grid)
end
