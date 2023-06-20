using Oceananigans.Operators: Δxᶜᵃᵃ, Δxᶠᵃᵃ, Δyᵃᶜᵃ, Δyᵃᶠᵃ, Δzᵃᵃᶜ, Δzᵃᵃᶠ
using Oceananigans.Grids: XYRegRectilinearGrid, XZRegRectilinearGrid, YZRegRectilinearGrid, stretched_dimensions
import Oceananigans.Architectures: architecture

struct FourierTridiagonalPoissonSolver{G, B, R, S, β, T, D}
                          grid :: G
    batched_tridiagonal_solver :: B
                   source_term :: R
                       storage :: S
                        buffer :: β
                    transforms :: T
         tridiagonal_direction :: D
end

architecture(solver::FourierTridiagonalPoissonSolver) = architecture(solver.grid)

@kernel function compute_main_diagonal!(D, grid, λy, λz, ::XDirection)
    j, k = @index(Global, NTuple)
    Nx = size(grid, 1)

    # Using a homogeneous Neumann (zero Gradient) boundary condition:
    D[1, j, k] = -1 / Δxᶠᵃᵃ(2, j, k, grid) - Δxᶜᵃᵃ(1, j, k, grid) * (λy[j] + λz[k])
    @unroll for i in 2:Nx-1
        D[i, j, k] = - (1 / Δxᶠᵃᵃ(i+1, j, k, grid) + 1 / Δxᶠᵃᵃ(i, j, k, grid)) - Δxᶜᵃᵃ(i, j, k, grid) * (λy[j] + λz[k])
    end
    D[Nx, j, k] = -1 / Δxᶠᵃᵃ(Nx, j, k, grid) - Δxᶜᵃᵃ(Nx, j, k, grid) * (λy[j] + λz[k])
end 

@kernel function compute_main_diagonal!(D, grid, λx, λz, ::YDirection)
    i, k = @index(Global, NTuple)
    Ny = size(grid, 2)

    # Using a homogeneous Neumann (zero Gradient) boundary condition:
    D[i, 1, k] = -1 / Δyᵃᶠᵃ(i, 2, k, grid) - Δyᵃᶜᵃ(i, 1, k, grid) * (λx[i] + λz[k])
    @unroll for j in 2:Ny-1
        D[i, j, k] = - (1 / Δyᵃᶠᵃ(i, j+1, k, grid) + 1 / Δyᵃᶠᵃ(i, j, k, grid)) - Δyᵃᶜᵃ(i, j, k, grid) * (λx[i] + λz[k])
    end
    D[i, Ny, k] = -1 / Δyᵃᶠᵃ(i, Ny, k, grid) - Δyᵃᶜᵃ(i, Ny, k, grid) * (λx[i] + λz[k])
end 

@kernel function compute_main_diagonal!(D, grid, λx, λy, ::ZDirection)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    # Using a homogeneous Neumann (zero Gradient) boundary condition:
    D[i, j, 1] = -1 / Δzᵃᵃᶠ(i, j, 2, grid) - Δzᵃᵃᶜ(i, j, 1, grid) * (λx[i] + λy[j])
    @unroll for k in 2:Nz-1
        D[i, j, k] = - (1 / Δzᵃᵃᶠ(i, j, k+1, grid) + 1 / Δzᵃᵃᶠ(i, j, k, grid)) - Δzᵃᵃᶜ(i, j, k, grid) * (λx[i] + λy[j])
    end
    D[i, j, Nz] = -1 / Δzᵃᵃᶠ(i, j, Nz, grid) - Δzᵃᵃᶜ(i, j, Nz, grid) * (λx[i] + λy[j])
end 


stretched_direction(::YZRegRectilinearGrid) = XDirection()
stretched_direction(::XZRegRectilinearGrid) = YDirection()
stretched_direction(::XYRegRectilinearGrid) = ZDirection()

Δξᶠ(i, grid::YZRegRectilinearGrid) = Δxᶠᵃᵃ(i, 1, 1, grid)
Δξᶠ(j, grid::XZRegRectilinearGrid) = Δyᵃᶠᵃ(1, j, 1, grid)
Δξᶠ(k, grid::XYRegRectilinearGrid) = Δzᵃᵃᶠ(1, 1, k, grid)

extent(grid) = (grid.Lx, grid.Ly, grid.Lz)

function FourierTridiagonalPoissonSolver(grid, planner_flag=FFTW.PATIENT)
    irreg_dim = stretched_dimensions(grid)[1]

    regular_top1, regular_top2 = Tuple( el for (i, el) in enumerate(topology(grid)) if i ≠ irreg_dim)
    regular_siz1, regular_siz2 = Tuple( el for (i, el) in enumerate(size(grid))     if i ≠ irreg_dim)
    regular_ext1, regular_ext2 = Tuple( el for (i, el) in enumerate(extent(grid))   if i ≠ irreg_dim)

    topology(grid, irreg_dim) != Bounded && error("`FourierTridiagonalPoissonSolver` can only be used when the stretched direction's topology is `Bounded`.")

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
    lower_diagonal = CUDA.@allowscalar [ 1 / Δξᶠ(q, grid) for q in 2:size(grid, irreg_dim) ]
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

    tridiagonal_direction = stretched_direction(grid)
    launch!(arch, grid, launch_config, compute_main_diagonal!, diagonal, grid, λ1, λ2, tridiagonal_direction)
    
    # Set up batched tridiagonal solver
    btsolver = BatchedTridiagonalSolver(grid; lower_diagonal, diagonal, upper_diagonal, tridiagonal_direction)

    # Need buffer for index permutations and transposes.
    buffer_needed = arch isa GPU && Bounded in (regular_top1, regular_top2)
    buffer = buffer_needed ? similar(sol_storage) : nothing

    # Storage space for right hand side of Poisson equation
    rhs = arch_array(arch, zeros(complex(eltype(grid)), size(grid)...))

    return FourierTridiagonalPoissonSolver(grid, btsolver, rhs, sol_storage, buffer, transforms, tridiagonal_direction)
end

function solve!(x, solver::FourierTridiagonalPoissonSolver, b=nothing)
    !isnothing(b) && set_source_term!(solver, b) # otherwise, assume source term is set correctly

    arch = architecture(solver)
    ϕ = solver.storage

    # Apply forward transforms in order
    [transform!(solver.source_term, solver.buffer) for transform! in solver.transforms.forward]

    # Solve tridiagonal system of linear equations at every column.
    solve!(ϕ, solver.batched_tridiagonal_solver, solver.source_term)

    # Apply backward transforms in order
    [transform!(ϕ, solver.buffer) for transform! in solver.transforms.backward]

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
multiplying it by the vertical grid spacing at cell centers in the stretched direction.
"""
function set_source_term!(solver::FourierTridiagonalPoissonSolver, source_term)
    grid = solver.grid
    arch = architecture(solver)
    solver.source_term .= source_term

    launch!(arch, grid, :xyz, multiply_by_stretched_spacing!, solver.source_term, grid)

    return nothing
end


@kernel function multiply_by_stretched_spacing!(a, grid::YZRegRectilinearGrid)
    i, j, k = @index(Global, NTuple)
    a[i, j, k] *= Δxᶜᵃᵃ(i, j, k, grid)
end

@kernel function multiply_by_stretched_spacing!(a, grid::XZRegRectilinearGrid)
    i, j, k = @index(Global, NTuple)
    a[i, j, k] *= Δyᵃᶜᵃ(i, j, k, grid)
end

@kernel function multiply_by_stretched_spacing!(a, grid::XYRegRectilinearGrid)
    i, j, k = @index(Global, NTuple)
    a[i, j, k] *= Δzᵃᵃᶜ(i, j, k, grid)
end
