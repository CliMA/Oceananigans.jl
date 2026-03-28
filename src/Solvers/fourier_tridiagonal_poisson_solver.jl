using Oceananigans.Operators: Δxᶜᶜᶜ, Δxᶜᵃᵃ, Δxᶠᵃᵃ, Δyᵃᶜᵃ, Δyᵃᶠᵃ, Δyᶜᶜᶜ, Δzᵃᵃᶜ, Δzᵃᵃᶠ, Δzᶜᶜᶜ
using Oceananigans.Grids: XYRegularRG, XZRegularRG, YZRegularRG, XYZRegularRG

struct FourierTridiagonalPoissonSolver{G, F, Λ, B, R, S, β, T}
    grid :: G
    tridiagonal_formulation :: F
    poisson_eigenvalues :: Λ
    batched_tridiagonal_solver :: B
    source_term :: R
    storage :: S
    buffer :: β
    transforms :: T
end

function Base.show(io::IO, solver::FourierTridiagonalPoissonSolver)
    print(io, "FourierTridiagonalPoissonSolver", '\n')
    print(io, "├── batched_tridiagonal_solver: ", prettysummary(solver.batched_tridiagonal_solver), '\n')
    print(io, "└── grid: ", prettysummary(solver.grid))
end

Architectures.architecture(solver::FourierTridiagonalPoissonSolver) = architecture(solver.grid)

stretched_direction(::YZRegularRG) = XDirection()
stretched_direction(::XZRegularRG) = YDirection()
stretched_direction(::XYRegularRG) = ZDirection()

dimension(::XDirection) = 1
dimension(::YDirection) = 2
dimension(::ZDirection) = 3

main_diagonal_launch_configuration(::XDirection) = :yz
main_diagonal_launch_configuration(::YDirection) = :xz
main_diagonal_launch_configuration(::ZDirection) = :xy

extent(grid) = (grid.Lx, grid.Ly, grid.Lz)

abstract type AbstractHomogeneousNeumannFormulation end
abstract type AbstractInhomogeneousNeumannFormulation end

struct InhomogeneousFormulation{D} <: AbstractInhomogeneousNeumannFormulation
    direction :: D
end

struct HomogeneousNeumannFormulation{D} <: AbstractHomogeneousNeumannFormulation
    direction :: D
end

tridiagonal_direction(formulation::HomogeneousNeumannFormulation) = formulation.direction
tridiagonal_direction(formulation::InhomogeneousFormulation) = formulation.direction

const HomogeneousXFormulation = HomogeneousNeumannFormulation{<:XDirection}
const HomogeneousYFormulation = HomogeneousNeumannFormulation{<:YDirection}
const HomogeneousZFormulation = HomogeneousNeumannFormulation{<:ZDirection}

const InhomogeneousXFormulation = InhomogeneousFormulation{<:XDirection}
const InhomogeneousYFormulation = InhomogeneousFormulation{<:YDirection}
const InhomogeneousZFormulation = InhomogeneousFormulation{<:ZDirection}

"""
    FourierTridiagonalPoissonSolver(grid, planner_flag = FFTW.PATIENT; tridiagonal_formulation=nothing)

Construct a `FourierTridiagonalPoissonSolver` on `grid` with `tridiagonal_formulation` either
`XDirection()`, `YDirection()`, or `ZDirection()`. The `tridiagonal_formulation` can be used to tweak
the tridiagonal matrices to solve variants on the Poisson equation, such as the screened Poisson equation,

```math
(∇² + m) ϕ = b
```

or the Poisson-like equation

```math
∂x² ϕ + ∂y² ϕ + ∂z (L ∂z ϕ) = b
```

or to implement boundary conditions other than the standard homogeneous Neumann boundary conditions.

The tridiagonal direction is determined by is `tridiagonal_direction(tridiagonal_formulation)`.

If `tridiagonal_formulation` is not specified, the tridiagonal direction is selected as the variably-spaced
direction of the grid, or as the `ZDirection()` for grids with uniform spacing in all three directions.

The (possibly perturbed) Poisson equation is solved with an FFT-based eigenfunction expansion in the non-tridiagonal-directions
augmented by a tridiagonal solve in the tridiagonal direction.
The non-tridiagonal-directions must be uniformly spaced.
"""
function FourierTridiagonalPoissonSolver(grid, planner_flag=FFTW.PATIENT; tridiagonal_formulation=nothing)

    # Try to guess what direction should be tridiagonal
    if isnothing(tridiagonal_formulation)
        tridiagonal_dir = grid isa XYZRegularRG ? ZDirection() : stretched_direction(grid)
        tridiagonal_formulation = HomogeneousNeumannFormulation(tridiagonal_dir)
    else
        tridiagonal_dir = tridiagonal_direction(tridiagonal_formulation)
    end

    tridiagonal_dim = dimension(tridiagonal_dir)

    if topology(grid, tridiagonal_dim) != Bounded
        msg = "`FourierTridiagonalPoissonSolver` can only be used \
                when the stretched direction's topology is `Bounded`."
        throw(ArgumentError(msg))
    end

    # Compute discrete Poisson eigenvalues
    N1, N2 = Tuple(el for (i, el) in enumerate(size(grid))     if i ≠ tridiagonal_dim)
    T1, T2 = Tuple(el for (i, el) in enumerate(topology(grid)) if i ≠ tridiagonal_dim)
    L1, L2 = Tuple(el for (i, el) in enumerate(extent(grid))   if i ≠ tridiagonal_dim)

    λ1 = poisson_eigenvalues(N1, L1, 1, T1())
    λ2 = poisson_eigenvalues(N2, L2, 2, T2())

    arch = architecture(grid)
    λ1 = on_architecture(arch, λ1)
    λ2 = on_architecture(arch, λ2)

    # Plan required transforms for x and y
    CT = complex(eltype(grid))
    sol_storage = on_architecture(arch, zeros(CT, size(grid)...))
    transforms = plan_transforms(grid, sol_storage, planner_flag, tridiagonal_dim)

    # Lower and upper diagonals are the same
    main_diagonal = zeros(grid, size(grid)...)

    Nd = size(grid, tridiagonal_dim) - 1
    lower_diagonal = zeros(grid, Nd)
    upper_diagonal = lower_diagonal

    compute_main_diagonal!(main_diagonal, tridiagonal_formulation, grid, λ1, λ2)
    Nd > 0 && compute_lower_diagonal!(lower_diagonal, tridiagonal_formulation, grid)

    # Set up batched tridiagonal solver
    btsolver = BatchedTridiagonalSolver(grid; lower_diagonal, upper_diagonal,
                                        diagonal = main_diagonal,
                                        tridiagonal_direction = tridiagonal_dir)

    # Need buffer for index permutations and transposes.
    buffer_needed = arch isa GPU && Bounded in (T1, T2)
    buffer = buffer_needed ? similar(sol_storage) : nothing

    # Storage space for right hand side of Poisson equation
    CT = complex(eltype(grid))
    rhs = on_architecture(arch, zeros(CT, size(grid)...))

    eigenvalues = (λ1, λ2)

    return FourierTridiagonalPoissonSolver(grid, tridiagonal_formulation, eigenvalues, btsolver,
                                           rhs, sol_storage, buffer, transforms)
end

#####
##### Setup utilities
#####

# Note: diagonal coefficients depend on non-tridiagonal directions because
# eigenvalues depend on non-tridiagonal directions.
function compute_main_diagonal!(main_diagonal, tridiagonal_formulation, grid, λ1, λ2)
    tridiagonal_dir = tridiagonal_direction(tridiagonal_formulation)
    launch_config = main_diagonal_launch_configuration(tridiagonal_dir)
    arch = grid.architecture
    launch!(arch, grid, launch_config, _compute_main_diagonal!, main_diagonal, grid, λ1, λ2, tridiagonal_formulation)
    return nothing
end

const XFormulation = Union{HomogeneousXFormulation, InhomogeneousXFormulation}
const YFormulation = Union{HomogeneousYFormulation, InhomogeneousYFormulation}
const ZFormulation = Union{HomogeneousZFormulation, InhomogeneousZFormulation}

@kernel function _compute_main_diagonal!(D, grid, λy, λz, ::XFormulation)
    j, k = @index(Global, NTuple)
    Nx = size(grid, 1)

    # Using a homogeneous Neumann (zero Gradient) boundary condition:
    @inbounds begin
        D[1, j, k]  = -1 / Δxᶠᵃᵃ( 2, j, k, grid) - Δxᶜᵃᵃ( 1, j, k, grid) * (λy[j] + λz[k])
        D[Nx, j, k] = -1 / Δxᶠᵃᵃ(Nx, j, k, grid) - Δxᶜᵃᵃ(Nx, j, k, grid) * (λy[j] + λz[k])

        for i in 2:Nx-1
            D[i, j, k] = - (1 / Δxᶠᵃᵃ(i+1, j, k, grid) + 1 / Δxᶠᵃᵃ(i, j, k, grid)) - Δxᶜᵃᵃ(i, j, k, grid) * (λy[j] + λz[k])
        end
    end
end

@kernel function _compute_main_diagonal!(D, grid, λx, λz, ::YFormulation)
    i, k = @index(Global, NTuple)
    Ny = size(grid, 2)

    # Using a homogeneous Neumann (zero Gradient) boundary condition:
    @inbounds begin
        D[i, 1, k]  = -1 / Δyᵃᶠᵃ(i,  2, k, grid) - Δyᵃᶜᵃ(i,  1, k, grid) * (λx[i] + λz[k])
        D[i, Ny, k] = -1 / Δyᵃᶠᵃ(i, Ny, k, grid) - Δyᵃᶜᵃ(i, Ny, k, grid) * (λx[i] + λz[k])

        for j in 2:Ny-1
            D[i, j, k] = - (1 / Δyᵃᶠᵃ(i, j+1, k, grid) + 1 / Δyᵃᶠᵃ(i, j, k, grid)) - Δyᵃᶜᵃ(i, j, k, grid) * (λx[i] + λz[k])
        end
    end
end

@kernel function _compute_main_diagonal!(D, grid, λx, λy, ::ZFormulation)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    # Using a homogeneous Neumann (zero Gradient) boundary condition:
    @inbounds begin
        D[i, j, 1]  = -1 / Δzᵃᵃᶠ(i, j,  2, grid) - Δzᵃᵃᶜ(i, j,  1, grid) * (λx[i] + λy[j])
        D[i, j, Nz] = -1 / Δzᵃᵃᶠ(i, j, Nz, grid) - Δzᵃᵃᶜ(i, j, Nz, grid) * (λx[i] + λy[j])

        for k in 2:Nz-1
            D[i, j, k] = - (1 / Δzᵃᵃᶠ(i, j, k+1, grid) + 1 / Δzᵃᵃᶠ(i, j, k, grid)) - Δzᵃᵃᶜ(i, j, k, grid) * (λx[i] + λy[j])
        end
    end
end

Δξᶠ(i, grid, ::XDirection) = Δxᶠᵃᵃ(i, 1, 1, grid)
Δξᶠ(j, grid, ::YDirection) = Δyᵃᶠᵃ(1, j, 1, grid)
Δξᶠ(k, grid, ::ZDirection) = Δzᵃᵃᶠ(1, 1, k, grid)

function compute_lower_diagonal!(lower_diagonal, tridiagonal_formulation, grid)
    N = length(lower_diagonal)
    arch = grid.architecture
    launch!(arch, grid, tuple(N), _compute_lower_diagonal!, lower_diagonal, tridiagonal_formulation, grid)
    return nothing
end

@kernel function _compute_lower_diagonal!(lower_diagonal, formulation, grid)
    q = @index(Global)
    dir = tridiagonal_direction(formulation)
    @inbounds lower_diagonal[q] = 1 / Δξᶠ(q+1, grid, dir)
end

function solve!(x, solver::FourierTridiagonalPoissonSolver, b=nothing)
    !isnothing(b) && set_source_term!(solver, b) # otherwise, assume source term is set correctly

    # Apply forward transforms in order
    for transform! in solver.transforms.forward
        transform!(solver.source_term, solver.buffer)
    end

    # Solve tridiagonal system of linear equations at every column.
    ϕ = solver.storage
    solve!(ϕ, solver.batched_tridiagonal_solver, solver.source_term)

    # Apply backward transforms in order
    for transform! in solver.transforms.backward
        transform!(ϕ, solver.buffer)
    end

    # Set the volume mean of the solution to be zero.
    # Solutions to Poisson's equation are only unique up to a constant (the global mean
    # of the solution), so we need to pick a constant. We choose the constant to be zero
    # so that the solution has zero-mean.
    if solver.tridiagonal_formulation isa AbstractHomogeneousNeumannFormulation
        ϕ .= ϕ .- sum(ϕ) / length(ϕ)
    end

    arch = architecture(solver)
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
    tdir = solver.batched_tridiagonal_solver.tridiagonal_direction
    launch!(arch, grid, :xyz, multiply_by_spacing!, solver.source_term, tdir, grid)
    return nothing
end

@kernel function multiply_by_spacing!(b, ::XDirection, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds b[i, j, k] *= Δxᶜᶜᶜ(i, j, k, grid)
end

@kernel function multiply_by_spacing!(b, ::YDirection, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds b[i, j, k] *= Δyᶜᶜᶜ(i, j, k, grid)
end

@kernel function multiply_by_spacing!(b, ::ZDirection, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds b[i, j, k] *= Δzᶜᶜᶜ(i, j, k, grid)
end
