using LinearAlgebra: SymTridiagonal, eigen, mul!

#####
##### Fourier-tridiagonal solvers for the free-surface pressure Poisson equation,
#####
#####     ∇²ϕ - ϕ/(den Δz) δ(k=Nz) = δ,    den = g Δt² + Δzᶠ/2,
#####
##### where the second term is the Robin boundary condition imposed by an implicit
##### free surface. On grids with uniform x and y the Robin term lives in the
##### tridiagonal (z) direction and is handled by `InhomogeneousFormulation(ZDirection())`.
##### On x- or y-stretched grids the tridiagonal direction is horizontal, so instead the
##### z transform diagonalizes the vertical operator *including* the Robin term: with
##### uniform z the per-volume vertical operator is the same symmetric tridiagonal matrix
##### in every column, and its eigenvectors replace the cosine basis while its (negated)
##### eigenvalues replace the staggered-Neumann Poisson eigenvalues. The eigenbasis
##### depends on Δt through `den` and is recomputed by `update_robin_eigenbasis!`
##### whenever Δt changes.
#####

"""
$(TYPEDSIGNATURES)

Denominator `den = g Δt² + Δzᶠ / 2` of the free-surface Robin boundary condition,
shared by the direct solvers, the CG linear operator, and the source terms.
"""
@inline robin_denominator(g, Δt, Δzᶠ) = g * Δt^2 + Δzᶠ / 2

struct RobinEigenbasisFormulation{D, M, B, T}
    direction :: D
    eigenvectors :: M
    buffer :: B
    Δt :: T
end

tridiagonal_direction(formulation::RobinEigenbasisFormulation) = formulation.direction

Base.summary(::RobinEigenbasisFormulation) = "RobinEigenbasisFormulation"

struct EigenbasisTransform{D, M, B}
    direction :: D
    eigenvectors :: M
    buffer :: B
end

(transform::EigenbasisTransform{<:Forward})(A, buffer) =
    apply_eigenbasis!(A, transform.buffer, transform.eigenvectors)

(transform::EigenbasisTransform{<:Backward})(A, buffer) =
    apply_eigenbasis!(A, transform.buffer, transpose(transform.eigenvectors))

function apply_eigenbasis!(A, buffer, Q)
    Nz = size(A, 3)
    A² = reshape(A, :, Nz)
    B² = reshape(buffer, :, Nz)
    mul!(B², A², Q)
    copyto!(A², B²)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Return the eigenvalues (negated, matching the sign convention of `poisson_eigenvalues`)
and eigenvectors of the one-dimensional vertical operator with `Nz` uniformly spaced
cells of height `Δz`, a homogeneous Neumann condition at the bottom, and the free-surface
Robin condition at the top.
"""
function robin_eigendecomposition(FT, Nz, Δz, gravitational_acceleration, Δt)
    g = Float64(gravitational_acceleration)
    Δz = Float64(Δz)
    Δt = Float64(Δt)
    den = robin_denominator(g, Δt, Δz)

    diagonal = [- ((k > 1) + (k < Nz)) / Δz^2 - (k == Nz) / (den * Δz) for k in 1:Nz]
    off_diagonal = fill(1 / Δz^2, Nz - 1)
    decomposition = eigen(SymTridiagonal(diagonal, off_diagonal))

    λ = convert.(FT, .- decomposition.values)
    Q = convert.(FT, decomposition.vectors)

    return λ, Q
end

"""
$(TYPEDSIGNATURES)

Refresh the vertical eigenbasis, Poisson eigenvalues, and tridiagonal main diagonal of a
`FourierTridiagonalPoissonSolver` with a `RobinEigenbasisFormulation` for a new time step
`Δt`. No-op when `Δt` matches the cached value.
"""
function update_robin_eigenbasis!(solver::FourierTridiagonalPoissonSolver, gravitational_acceleration, Δt)
    formulation = solver.tridiagonal_formulation
    Δt == formulation.Δt[] && return nothing
    formulation.Δt[] = Δt

    grid = solver.grid
    FT = eltype(grid)
    Nz = size(grid, 3)
    Δz = Δzᵃᵃᶜ(1, 1, 1, grid)
    λz, Q = robin_eigendecomposition(FT, Nz, Δz, gravitational_acceleration, Δt)

    λ1, λ2 = solver.poisson_eigenvalues
    copyto!(λ2, reshape(λz, size(λ2)))
    copyto!(formulation.eigenvectors, convert.(eltype(formulation.eigenvectors), Q))

    neumann_formulation = HomogeneousNeumannFormulation(formulation.direction)
    main_diagonal = solver.batched_tridiagonal_solver.b
    compute_main_diagonal!(main_diagonal, neumann_formulation, grid, λ1, λ2)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Return a `FourierTridiagonalPoissonSolver` for the pressure Poisson equation with a
free-surface Robin boundary condition on `grid`. On grids with uniform x and y spacing
the Robin condition is imposed through `InhomogeneousFormulation(ZDirection())`; on
x- or y-stretched grids it is imposed through a `RobinEigenbasisFormulation`, which
diagonalizes the vertical direction with the eigenbasis of the Robin vertical operator.
"""
fourier_tridiagonal_free_surface_solver(grid::XYZRegularRG) =
    FourierTridiagonalPoissonSolver(grid; tridiagonal_formulation=InhomogeneousFormulation(ZDirection()))

fourier_tridiagonal_free_surface_solver(grid::XYRegularRG) =
    FourierTridiagonalPoissonSolver(grid; tridiagonal_formulation=InhomogeneousFormulation(ZDirection()))

fourier_tridiagonal_free_surface_solver(grid::XZRegularRG) = robin_eigenbasis_poisson_solver(grid)
fourier_tridiagonal_free_surface_solver(grid::YZRegularRG) = robin_eigenbasis_poisson_solver(grid)

function robin_eigenbasis_poisson_solver(grid, planner_flag=FFTW.PATIENT)
    topology(grid, 3) === Bounded ||
        throw(ArgumentError("`RobinEigenbasisFormulation` requires a `Bounded` z topology."))

    tridiagonal_dir = stretched_direction(grid)
    tridiagonal_dim = dimension(tridiagonal_dir)
    transform_dim = tridiagonal_dim == 1 ? 2 : 1

    topology(grid, tridiagonal_dim) === Bounded ||
        throw(ArgumentError("`FourierTridiagonalPoissonSolver` can only be used " *
                            "when the stretched direction's topology is `Bounded`."))

    arch = architecture(grid)
    FT = eltype(grid)
    Nz = size(grid, 3)

    T1 = topology(grid, transform_dim)
    λ1 = poisson_eigenvalues(grid, size(grid, transform_dim), extent(grid)[transform_dim], 1, T1())
    λ1 = on_architecture(arch, λ1)

    # Placeholder z eigenvalues; `update_robin_eigenbasis!` fills them before the first solve.
    λ2 = on_architecture(arch, zeros(FT, reshaped_size(Nz, 2)...))

    CT = complex(FT)
    sol_storage = on_architecture(arch, zeros(CT, size(grid)...))

    # On the GPU, dim-2 transforms run on a transposed copy (`DiscreteTransform` sets
    # `transpose_dims`), so their plans must be built on the (Ny, Nx, Nz) layout.
    if arch isa GPU && transform_dim == 2
        Nx, Ny, _ = size(grid)
        plan_storage = reshape(sol_storage, (Ny, Nx, Nz))
        plan_dims = [1]
    else
        plan_storage = sol_storage
        plan_dims = [transform_dim]
    end

    forward_plan = plan_forward_transform(plan_storage, T1(), plan_dims, planner_flag)
    backward_plan = plan_backward_transform(plan_storage, T1(), plan_dims, planner_flag)
    forward_horizontal = DiscreteTransform(forward_plan, Forward(), grid, [transform_dim])
    backward_horizontal = DiscreteTransform(backward_plan, Backward(), grid, [transform_dim])

    eigenvectors = on_architecture(arch, zeros(CT, Nz, Nz))
    eigenbasis_buffer = similar(sol_storage)
    formulation = RobinEigenbasisFormulation(tridiagonal_dir, eigenvectors, eigenbasis_buffer, Ref(convert(FT, NaN)))

    # The bounded horizontal transform (a GPU DCT) must run before the eigenbasis
    # transform going forward, and after it going backward.
    transforms = (forward = (forward_horizontal, EigenbasisTransform(Forward(), eigenvectors, eigenbasis_buffer)),
                  backward = (EigenbasisTransform(Backward(), eigenvectors, eigenbasis_buffer), backward_horizontal))

    main_diagonal = zeros(grid, size(grid)...)
    Nd = size(grid, tridiagonal_dim) - 1
    lower_diagonal = zeros(grid, Nd)
    upper_diagonal = lower_diagonal

    neumann_formulation = HomogeneousNeumannFormulation(tridiagonal_dir)
    compute_main_diagonal!(main_diagonal, neumann_formulation, grid, λ1, λ2)
    Nd > 0 && compute_lower_diagonal!(lower_diagonal, neumann_formulation, grid)

    btsolver = BatchedTridiagonalSolver(grid; lower_diagonal, upper_diagonal,
                                        diagonal = main_diagonal,
                                        tridiagonal_direction = tridiagonal_dir)

    # The buffer serves GPU index permutation (`Bounded`) and the dim-2 transpose (any topology).
    buffer_needed = arch isa GPU && T1 !== Flat && (T1 === Bounded || transform_dim == 2)
    buffer = buffer_needed ? similar(sol_storage) : nothing

    rhs = on_architecture(arch, zeros(CT, size(grid)...))

    return FourierTridiagonalPoissonSolver(grid, formulation, (λ1, λ2), btsolver,
                                           rhs, sol_storage, buffer, transforms)
end
