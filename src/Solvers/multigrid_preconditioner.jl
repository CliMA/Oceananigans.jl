using Oceananigans.Utils: KernelParameters

#####
##### A geometric multigrid preconditioner for the ConjugateGradientPoissonSolver
#####
##### The symmetric volume-weighted Laplacian V∇² is represented in "conductance form": each
##### face carries a conductance C = (face area) / (center-to-center distance), zeroed across
##### immersed and domain boundaries, so that
#####
#####     (V∇²ϕ)ᵢⱼₖ = Σ_faces C (ϕ_neighbor - ϕ) .
#####
##### Coarse levels agglomerate cells in the horizontal only (never in z) and sum the fine
##### conductances crossing each coarse face, which is the Galerkin operator RAP for
##### piecewise-constant transfers. Vertical stretching, immersed boundaries, and partial cells
##### therefore propagate to all levels through the coefficients alone. The smoother is
##### red-black line relaxation that solves each vertical column exactly with a Thomas sweep,
##### which is the appropriate smoother for large-aspect-ratio (Δz ≪ Δx) ocean grids.
#####

struct MultigridLevel{A, E}
    Cx :: A
    Cy :: A
    Cz :: A
    D  :: A
    E  :: E
    ϕ  :: A
    b  :: A
    r  :: A
    t  :: A
    coarsen_x :: Bool
    coarsen_y :: Bool
end

Base.size(level::MultigridLevel) = size(level.D)

struct MultigridPreconditioner{G, L, T}
    grid :: G
    levels :: Vector{L}
    presmoothing_sweeps :: Int
    postsmoothing_sweeps :: Int
    coarsest_sweeps :: Int
    regularization :: T
end

Base.summary(mg::MultigridPreconditioner) = "MultigridPreconditioner with $(length(mg.levels)) levels"

function Base.show(io::IO, mg::MultigridPreconditioner)
    print(io, summary(mg))
    for (ℓ, level) in enumerate(mg.levels)
        nx, ny, nz = size(level)
        connector = ℓ == length(mg.levels) ? "└──" : "├──"
        print(io, '\n', connector, " level ", ℓ, ": ", nx, "×", ny, "×", nz)
    end
end

#####
##### Fine-level conductances from grid metrics
#####

@kernel function _fine_x_conductance!(Cx, grid, periodic, Nx)
    i, j, k = @index(Global, NTuple)
    boundary = (i == 1) | (i == Nx + 1)
    iˡ = ifelse(i == 1, Nx, i - 1)
    iʳ = ifelse(i == Nx + 1, 1, i)
    active = !(inactive_cell(iˡ, j, k, grid) | inactive_cell(iʳ, j, k, grid))
    c = Axᶠᶜᶜ(i, j, k, grid) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    @inbounds Cx[i, j, k] = ifelse((boundary & !periodic) | !active, zero(grid), c)
end

@kernel function _fine_y_conductance!(Cy, grid, periodic, Ny)
    i, j, k = @index(Global, NTuple)
    boundary = (j == 1) | (j == Ny + 1)
    jˡ = ifelse(j == 1, Ny, j - 1)
    jʳ = ifelse(j == Ny + 1, 1, j)
    active = !(inactive_cell(i, jˡ, k, grid) | inactive_cell(i, jʳ, k, grid))
    c = Ayᶜᶠᶜ(i, j, k, grid) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    @inbounds Cy[i, j, k] = ifelse((boundary & !periodic) | !active, zero(grid), c)
end

@kernel function _fine_z_conductance!(Cz, grid, Nz)
    i, j, k = @index(Global, NTuple)
    boundary = (k == 1) | (k == Nz + 1)
    active = !(inactive_cell(i, j, k - 1, grid) | inactive_cell(i, j, k, grid))
    c = Azᶜᶜᶠ(i, j, k, grid) * Δz⁻¹ᶜᶜᶠ(i, j, k, grid)
    @inbounds Cz[i, j, k] = ifelse(boundary | !active, zero(grid), c)
end

#####
##### Galerkin coarsening: sum the fine conductances crossing each coarse face
#####

@kernel function _coarsen_x_conductance!(Cxᶜ, Cxᶠ, cx, cy, Nxᶠ, Nyᶠ)
    I, J, k = @index(Global, NTuple)
    f  = ifelse(cx, min(2I - 1, Nxᶠ + 1), I)
    j₁ = ifelse(cy, 2J - 1, J)
    j₂ = ifelse(cy, min(2J, Nyᶠ), J)
    s = zero(eltype(Cxᶜ))
    @inbounds for j in j₁:j₂
        s += Cxᶠ[f, j, k]
    end
    @inbounds Cxᶜ[I, J, k] = s
end

@kernel function _coarsen_y_conductance!(Cyᶜ, Cyᶠ, cx, cy, Nxᶠ, Nyᶠ)
    I, J, k = @index(Global, NTuple)
    f  = ifelse(cy, min(2J - 1, Nyᶠ + 1), J)
    i₁ = ifelse(cx, 2I - 1, I)
    i₂ = ifelse(cx, min(2I, Nxᶠ), I)
    s = zero(eltype(Cyᶜ))
    @inbounds for i in i₁:i₂
        s += Cyᶠ[i, f, k]
    end
    @inbounds Cyᶜ[I, J, k] = s
end

@kernel function _coarsen_z_conductance!(Czᶜ, Czᶠ, cx, cy, Nxᶠ, Nyᶠ)
    I, J, k = @index(Global, NTuple)
    i₁ = ifelse(cx, 2I - 1, I)
    i₂ = ifelse(cx, min(2I, Nxᶠ), I)
    j₁ = ifelse(cy, 2J - 1, J)
    j₂ = ifelse(cy, min(2J, Nyᶠ), J)
    s = zero(eltype(Czᶜ))
    @inbounds for j in j₁:j₂, i in i₁:i₂
        s += Czᶠ[i, j, k]
    end
    @inbounds Czᶜ[I, J, k] = s
end

# Cells with no conductances (immersed or isolated) get D = -1 so the smoother acts as the
# identity there; their right-hand side is always zero so the correction stays zero.
@kernel function _compute_diagonal!(D, Cx, Cy, Cz)
    i, j, k = @index(Global, NTuple)
    @inbounds s = -(Cx[i, j, k] + Cx[i+1, j, k] +
                    Cy[i, j, k] + Cy[i, j+1, k] +
                    Cz[i, j, k] + Cz[i, j, k+1])
    @inbounds D[i, j, k] = ifelse(s == 0, -one(s), s)
end

# A column with no horizontal conductance anywhere is a sealed Neumann sub-system whose
# tridiagonal is singular; only there is the diagonal shifted by ε. Coupled columns are
# strictly diagonally dominant, and even a tiny uniform shift would drown the horizontal
# couplings (which can be ~10⁻⁶ of the diagonal on strongly anisotropic grids) that the
# coarse-grid correction relies on.
@kernel function _compute_column_regularization!(E, Cx, Cy, Nz, ε)
    i, j = @index(Global, NTuple)
    h = zero(eltype(E))
    @inbounds for k in 1:Nz
        h += Cx[i, j, k] + Cx[i+1, j, k] + Cy[i, j, k] + Cy[i, j+1, k]
    end
    @inbounds E[i, j] = ifelse(h == 0, ε, zero(ε))
end

#####
##### Level operations: residual, restriction, prolongation, smoothing
#####

@kernel function _compute_level_residual!(r, ϕ, b, Cx, Cy, Cz, D, Nx, Ny, Nz)
    i, j, k = @index(Global, NTuple)
    i⁻ = ifelse(i == 1, Nx, i - 1)
    i⁺ = ifelse(i == Nx, 1, i + 1)
    j⁻ = ifelse(j == 1, Ny, j - 1)
    j⁺ = ifelse(j == Ny, 1, j + 1)
    k⁻ = max(k - 1, 1)
    k⁺ = min(k + 1, Nz)
    @inbounds r[i, j, k] = b[i, j, k] - (D[i, j, k]    * ϕ[i, j, k] +
                                         Cx[i, j, k]   * ϕ[i⁻, j, k] + Cx[i+1, j, k] * ϕ[i⁺, j, k] +
                                         Cy[i, j, k]   * ϕ[i, j⁻, k] + Cy[i, j+1, k] * ϕ[i, j⁺, k] +
                                         Cz[i, j, k]   * ϕ[i, j, k⁻] + Cz[i, j, k+1] * ϕ[i, j, k⁺])
end

@kernel function _restrict_residual!(bᶜ, r, cx, cy, Nxᶠ, Nyᶠ)
    I, J, k = @index(Global, NTuple)
    i₁ = ifelse(cx, 2I - 1, I)
    i₂ = ifelse(cx, min(2I, Nxᶠ), I)
    j₁ = ifelse(cy, 2J - 1, J)
    j₂ = ifelse(cy, min(2J, Nyᶠ), J)
    s = zero(eltype(bᶜ))
    @inbounds for j in j₁:j₂, i in i₁:i₂
        s += r[i, j, k]
    end
    @inbounds bᶜ[I, J, k] = s
end

@kernel function _prolong_and_correct!(ϕᶠ, ϕᶜ, cx, cy)
    i, j, k = @index(Global, NTuple)
    I = ifelse(cx, (i + 1) >> 1, i)
    J = ifelse(cy, (j + 1) >> 1, j)
    @inbounds ϕᶠ[i, j, k] += ϕᶜ[I, J, k]
end

# Red-black line relaxation: for each column (i, j) of the given color, solve the vertical
# tridiagonal sub-system exactly (Thomas algorithm, cprime stored in t) with the horizontal
# couplings moved to the right-hand side using the current iterate.
@kernel function _smooth_columns!(ϕ, t, b, Cx, Cy, Cz, D, E, color, Nx, Ny, Nz)
    i, j = @index(Global, NTuple)
    if (i + j) % 2 == color
        i⁻ = ifelse(i == 1, Nx, i - 1)
        i⁺ = ifelse(i == Nx, 1, i + 1)
        j⁻ = ifelse(j == 1, Ny, j - 1)
        j⁺ = ifelse(j == Ny, 1, j + 1)

        @inbounds begin
            ε = E[i, j]

            β = D[i, j, 1] * (1 + ε)
            rhs = b[i, j, 1] - (Cx[i, j, 1] * ϕ[i⁻, j, 1] + Cx[i+1, j, 1] * ϕ[i⁺, j, 1] +
                                Cy[i, j, 1] * ϕ[i, j⁻, 1] + Cy[i, j+1, 1] * ϕ[i, j⁺, 1])
            ϕ[i, j, 1] = rhs / β

            for k in 2:Nz
                t[i, j, k] = Cz[i, j, k] / β
                β = D[i, j, k] * (1 + ε) - Cz[i, j, k] * t[i, j, k]
                rhs = b[i, j, k] - (Cx[i, j, k] * ϕ[i⁻, j, k] + Cx[i+1, j, k] * ϕ[i⁺, j, k] +
                                    Cy[i, j, k] * ϕ[i, j⁻, k] + Cy[i, j+1, k] * ϕ[i, j⁺, k])
                ϕ[i, j, k] = (rhs - Cz[i, j, k] * ϕ[i, j, k-1]) / β
            end

            for k in Nz-1:-1:1
                ϕ[i, j, k] -= t[i, j, k+1] * ϕ[i, j, k+1]
            end
        end
    end
end

@kernel function _array_from_field!(a, f)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = f[i, j, k]
end

@kernel function _field_from_array!(f, a)
    i, j, k = @index(Global, NTuple)
    @inbounds f[i, j, k] = a[i, j, k]
end

#####
##### Construction
#####

function allocate_multigrid_level(arch, FT, nx, ny, nz, cx, cy)
    level_array(dims...) = on_architecture(arch, zeros(FT, dims...))
    return MultigridLevel(level_array(nx + 1, ny, nz),
                          level_array(nx, ny + 1, nz),
                          level_array(nx, ny, nz + 1),
                          level_array(nx, ny, nz),
                          level_array(nx, ny),
                          level_array(nx, ny, nz),
                          level_array(nx, ny, nz),
                          level_array(nx, ny, nz),
                          level_array(nx, ny, nz),
                          cx, cy)
end

"""
    MultigridPreconditioner(grid;
                            maxlevels = 99,
                            presmoothing_sweeps = 1,
                            postsmoothing_sweeps = 1,
                            coarsest_sweeps = 4)

Construct a geometric multigrid preconditioner for the [`ConjugateGradientPoissonSolver`](@ref)
that approximates `(V∇²)⁻¹` with one V-cycle per application.

The symmetric volume-weighted Laplacian `V∇²` is stored in conductance form (face area ×
inverse center-to-center distance, zeroed across immersed and domain boundaries). Coarse
levels are built by agglomerating cells `2 × 2` in the horizontal — never in the vertical —
and summing the fine-level conductances that cross each coarse face, which is the Galerkin
coarse operator for piecewise-constant transfers. Immersed boundaries, partial cells, and
grid stretching in any direction therefore enter every level through the coefficients, and
no FFT-solvability of the underlying grid is required: the preconditioner applies to grids
stretched in one, two, or three directions.

The smoother is red-black line relaxation that solves each vertical column exactly with a
Thomas sweep, so strong vertical grid anisotropy (`Δz ≪ Δx`) and vertical stretching are
absorbed by the smoother rather than degrading the multigrid convergence rate. Coarsening
continues (per direction, for non-`Flat` directions with at least 4 cells) until both
horizontal extents are smaller than 4 or `maxlevels` is reached; grid sizes need not be
powers of two. On the coarsest level the smoother is applied `coarsest_sweeps` times in a
symmetric red-black/black-red pattern, keeping the preconditioner symmetric as conjugate
gradient iteration requires.

Example
=======

```jldoctest
using Oceananigans
using Oceananigans.Solvers: MultigridPreconditioner

grid = RectilinearGrid(size=(16, 16, 8), extent=(1, 1, 1))
preconditioner = MultigridPreconditioner(grid)

# output
MultigridPreconditioner with 4 levels
├── level 1: 16×16×8
├── level 2: 8×8×8
├── level 3: 4×4×8
└── level 4: 2×2×8
```
"""
function MultigridPreconditioner(grid::AbstractGrid;
                                 maxlevels = 99,
                                 presmoothing_sweeps = 1,
                                 postsmoothing_sweeps = 1,
                                 coarsest_sweeps = 4)

    TX, TY, TZ = topology(grid)
    TZ === Bounded ||
        throw(ArgumentError("MultigridPreconditioner requires a Bounded z-direction (got $TZ)"))

    arch = architecture(grid)
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)

    sizes = [(Nx, Ny)]
    coarsenings = NTuple{2, Bool}[]
    while length(sizes) < maxlevels
        nx, ny = last(sizes)
        cx = TX !== Flat && nx >= 4
        cy = TY !== Flat && ny >= 4
        (cx || cy) || break
        push!(coarsenings, (cx, cy))
        push!(sizes, (cx ? cld(nx, 2) : nx, cy ? cld(ny, 2) : ny))
    end
    push!(coarsenings, (false, false))

    levels = [allocate_multigrid_level(arch, FT, nx, ny, Nz, cx, cy)
              for ((nx, ny), (cx, cy)) in zip(sizes, coarsenings)]

    fine = first(levels)
    TX === Flat ||
        launch!(arch, grid, KernelParameters((Nx + 1, Ny, Nz), (0, 0, 0)),
                _fine_x_conductance!, fine.Cx, grid, TX === Periodic, Nx)
    TY === Flat ||
        launch!(arch, grid, KernelParameters((Nx, Ny + 1, Nz), (0, 0, 0)),
                _fine_y_conductance!, fine.Cy, grid, TY === Periodic, Ny)
    launch!(arch, grid, KernelParameters((Nx, Ny, Nz + 1), (0, 0, 0)),
            _fine_z_conductance!, fine.Cz, grid, Nz)

    for ℓ in 1:length(levels)-1
        levelᶠ, levelᶜ = levels[ℓ], levels[ℓ+1]
        nxᶠ, nyᶠ, _ = size(levelᶠ)
        nxᶜ, nyᶜ, _ = size(levelᶜ)
        cx, cy = levelᶠ.coarsen_x, levelᶠ.coarsen_y
        launch!(arch, grid, KernelParameters((nxᶜ + 1, nyᶜ, Nz), (0, 0, 0)),
                _coarsen_x_conductance!, levelᶜ.Cx, levelᶠ.Cx, cx, cy, nxᶠ, nyᶠ)
        launch!(arch, grid, KernelParameters((nxᶜ, nyᶜ + 1, Nz), (0, 0, 0)),
                _coarsen_y_conductance!, levelᶜ.Cy, levelᶠ.Cy, cx, cy, nxᶠ, nyᶠ)
        launch!(arch, grid, KernelParameters((nxᶜ, nyᶜ, Nz + 1), (0, 0, 0)),
                _coarsen_z_conductance!, levelᶜ.Cz, levelᶠ.Cz, cx, cy, nxᶠ, nyᶠ)
    end

    ε = convert(FT, 1//100)
    for level in levels
        nx, ny, nz = size(level)
        launch!(arch, grid, KernelParameters((nx, ny, nz), (0, 0, 0)),
                _compute_diagonal!, level.D, level.Cx, level.Cy, level.Cz)
        launch!(arch, grid, KernelParameters((nx, ny), (0, 0)),
                _compute_column_regularization!, level.E, level.Cx, level.Cy, nz, ε)
    end

    return MultigridPreconditioner(grid, levels, presmoothing_sweeps, postsmoothing_sweeps,
                                   coarsest_sweeps, ε)
end

#####
##### The V-cycle
#####

function smooth_level!(mg::MultigridPreconditioner, level, first_color, second_color)
    grid = mg.grid
    arch = architecture(grid)
    nx, ny, nz = size(level)
    for color in (first_color, second_color)
        launch!(arch, grid, KernelParameters((nx, ny), (0, 0)),
                _smooth_columns!, level.ϕ, level.t, level.b, level.Cx, level.Cy, level.Cz,
                level.D, level.E, color, nx, ny, nz)
    end
    return nothing
end

function vcycle!(mg::MultigridPreconditioner, ℓ)
    grid = mg.grid
    arch = architecture(grid)
    level = mg.levels[ℓ]
    nx, ny, nz = size(level)

    fill!(level.ϕ, zero(eltype(level.ϕ)))

    if ℓ == length(mg.levels)
        # palindromic red-black/black-red sweeps keep the coarsest solve symmetric
        for _ in 1:mg.coarsest_sweeps
            smooth_level!(mg, level, 0, 1)
            smooth_level!(mg, level, 1, 0)
        end
        return nothing
    end

    for _ in 1:mg.presmoothing_sweeps
        smooth_level!(mg, level, 0, 1)
    end

    launch!(arch, grid, KernelParameters((nx, ny, nz), (0, 0, 0)),
            _compute_level_residual!, level.r, level.ϕ, level.b,
            level.Cx, level.Cy, level.Cz, level.D, nx, ny, nz)

    levelᶜ = mg.levels[ℓ+1]
    nxᶜ, nyᶜ, _ = size(levelᶜ)
    launch!(arch, grid, KernelParameters((nxᶜ, nyᶜ, nz), (0, 0, 0)),
            _restrict_residual!, levelᶜ.b, level.r, level.coarsen_x, level.coarsen_y, nx, ny)

    vcycle!(mg, ℓ + 1)

    launch!(arch, grid, KernelParameters((nx, ny, nz), (0, 0, 0)),
            _prolong_and_correct!, level.ϕ, levelᶜ.ϕ, level.coarsen_x, level.coarsen_y)

    for _ in 1:mg.postsmoothing_sweeps
        smooth_level!(mg, level, 1, 0)
    end

    return nothing
end

@inline function precondition!(z, mg::MultigridPreconditioner, r, args...)
    grid = mg.grid
    arch = architecture(grid)
    fine = first(mg.levels)
    launch!(arch, grid, :xyz, _array_from_field!, fine.b, r)
    vcycle!(mg, 1)
    launch!(arch, grid, :xyz, _field_from_array!, z, fine.ϕ)
    return z
end
