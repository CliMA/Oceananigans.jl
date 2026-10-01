using LinearAlgebra: pinv, mul!

#####
##### A geometric multigrid preconditioner for the ConjugateGradientPoissonSolver
#####
##### The symmetric volume-weighted Laplacian V∇² is stored in "conductance form": each face
##### carries a conductance C = (face area) / (center-to-center distance), zeroed across immersed
##### and domain boundaries, so that (V∇²ϕ)ᵢⱼₖ = Σ_faces C (ϕ_neighbor - ϕ).
#####

struct MultigridLevel{A, E}
    Cx :: A
    Cy :: A
    Cz :: A
    D  :: A
    E  :: E
    T  :: E
    β⁻¹ :: A
    t  :: A
    ϕ  :: A
    b  :: A
    r  :: A
    coarsen_x :: Bool
    coarsen_y :: Bool
end

Base.size(level::MultigridLevel) = size(level.D)

struct MultigridPreconditioner{G, L, M, S}
    grid :: G
    levels :: Vector{L}
    coarse_inverse :: M
    smoothing_sweeps :: Int
    cached_free_surface_timestep :: Base.RefValue{S}
end

cycle_float_type(mg::MultigridPreconditioner) = eltype(first(mg.levels).D)

function Base.summary(mg::MultigridPreconditioner)
    FT = cycle_float_type(mg)
    levels = "MultigridPreconditioner with $(length(mg.levels)) levels"
    return FT === eltype(mg.grid) ? levels : string(levels, " (", FT, " cycle)")
end

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
##### Coarsening: sum the fine conductances crossing each coarse face, halved in a coarsened
##### direction for the doubled center-to-center distance
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
    @inbounds Cxᶜ[I, J, k] = ifelse(cx, s / 2, s)
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
    @inbounds Cyᶜ[I, J, k] = ifelse(cy, s / 2, s)
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

# Cells with no conductances (immersed or isolated) get D = 1, so the smoother keeps them at zero
@kernel function _compute_diagonal!(D, Cx, Cy, Cz)
    i, j, k = @index(Global, NTuple)
    @inbounds s = -(Cx[i, j, k] + Cx[i+1, j, k] +
                    Cy[i, j, k] + Cy[i, j+1, k] +
                    Cz[i, j, k] + Cz[i, j, k+1])
    @inbounds D[i, j, k] = ifelse(s == 0, one(s), s)
end

# A column with no horizontal conductance is a singular Neumann sub-system whose diagonal is
# shifted by ε. Coupled columns are diagonally dominant only by their horizontal conductances,
# which can be ~10⁻⁶ of the diagonal on ocean grids; the floor εᶠ ~ Nz·eps keeps every Thomas
# pivot above rounding noise in Float32.
@kernel function _compute_column_regularization!(E, Cx, Cy, Nz, ε, εᶠ)
    i, j = @index(Global, NTuple)
    h = zero(eltype(E))
    @inbounds for k in 1:Nz
        h += Cx[i, j, k] + Cx[i+1, j, k] + Cy[i, j, k] + Cy[i, j+1, k]
    end
    @inbounds E[i, j] = ifelse(h == 0, ε, εᶠ)
end

#####
##### Level operations: residual, transfers, smoothing
#####

@kernel function _compute_level_residual!(r, ϕ, b, Cx, Cy, Cz, D, T, Nx, Ny, Nz)
    i, j, k = @index(Global, NTuple)
    i⁻ = ifelse(i == 1, Nx, i - 1)
    i⁺ = ifelse(i == Nx, 1, i + 1)
    j⁻ = ifelse(j == 1, Ny, j - 1)
    j⁺ = ifelse(j == Ny, 1, j + 1)
    k⁻ = max(k - 1, 1)
    k⁺ = min(k + 1, Nz)
    @inbounds Dᵏ = D[i, j, k] - ifelse(k == Nz, T[i, j], zero(eltype(T)))
    @inbounds active = D[i, j, k] < 0
    @inbounds r[i, j, k] = active * (b[i, j, k] - (Dᵏ            * ϕ[i, j, k] +
                                                   Cx[i, j, k]   * ϕ[i⁻, j, k] + Cx[i+1, j, k] * ϕ[i⁺, j, k] +
                                                   Cy[i, j, k]   * ϕ[i, j⁻, k] + Cy[i, j+1, k] * ϕ[i, j⁺, k] +
                                                   Cz[i, j, k]   * ϕ[i, j, k⁻] + Cz[i, j, k+1] * ϕ[i, j, k⁺]))
end

# In a coarsened direction, fine cell i interpolates between its parent coarse cell (weight 3/4)
# and the neighboring coarse cell on its other side (weight 1/4); in a direction that is not
# coarsened its parent is the cell itself.
@inline parent_index(i, coarsened) = ifelse(coarsened, (i + 1) >> 1, i)

# 0 when there is no neighbor beyond a non-periodic boundary
@inline function neighbor_index(i, nᶜ, coarsened, periodic)
    Iₙ = ifelse(isodd(i), (i + 1) >> 1 - 1, (i + 1) >> 1 + 1)
    Iₙ = ifelse(periodic & (Iₙ == 0), nᶜ, Iₙ)
    Iₙ = ifelse(periodic & (Iₙ == nᶜ + 1), 1, Iₙ)
    return ifelse(coarsened & (1 <= Iₙ <= nᶜ), Iₙ, 0)
end

# The weights of inactive or absent neighbors are given to the parent, so the weights sum to one
# and the correction extrapolates across immersed and domain boundaries.
@inline function interpolation_weights(Dᶜ, i, j, k, nxᶜ, nyᶜ, cx, cy, px, py)
    FT = eltype(Dᶜ)
    I = parent_index(i, cx)
    J = parent_index(j, cy)
    Iₙ = neighbor_index(i, nxᶜ, cx, px)
    Jₙ = neighbor_index(j, nyᶜ, cy, py)
    ax = ifelse(cx, FT(1//4), zero(FT))
    ay = ifelse(cy, FT(1//4), zero(FT))
    @inbounds begin
        wx  = ax * (1 - ay) * (Iₙ > 0) * (Dᶜ[max(Iₙ, 1), J, k] < 0)
        wy  = (1 - ax) * ay * (Jₙ > 0) * (Dᶜ[I, max(Jₙ, 1), k] < 0)
        wxy = ax * ay * (Iₙ > 0) * (Jₙ > 0) * (Dᶜ[max(Iₙ, 1), max(Jₙ, 1), k] < 0)
    end
    # absent neighbors have zero weight, so their index is only made valid
    return I, J, max(Iₙ, 1), max(Jₙ, 1), wx, wy, wxy
end

@kernel function _prolong_and_correct!(ϕᶠ, ϕᶜ, Dᶜ, Dᶠ, nxᶜ, nyᶜ, cx, cy, px, py)
    i, j, k = @index(Global, NTuple)
    I, J, Iₙ, Jₙ, wx, wy, wxy = interpolation_weights(Dᶜ, i, j, k, nxᶜ, nyᶜ, cx, cy, px, py)
    @inbounds begin
        active = Dᶠ[i, j, k] < 0
        ϕᶠ[i, j, k] += active * ((1 - wx - wy - wxy) * ϕᶜ[I, J, k] + wx  * ϕᶜ[Iₙ, J, k] +
                                 wy                  * ϕᶜ[I, Jₙ, k] + wxy * ϕᶜ[Iₙ, Jₙ, k])
    end
end

# The fine cells whose interpolation stencil can include coarse cell I: its two children and the
# cell just outside each; 0 marks a cell beyond a non-periodic boundary.
@inline function fine_index(I, m, nᶠ, coarsened, periodic)
    i = ifelse(coarsened, 2I - 3 + m, I)
    i = ifelse(periodic & (i < 1), i + nᶠ, i)
    i = ifelse(periodic & (i > nᶠ), i - nᶠ, i)
    return ifelse((1 <= i <= nᶠ) & (coarsened | (m == 1)), i, 0)
end

# Transpose of `_prolong_and_correct!`
@kernel function _restrict_residual!(bᶜ, r, Dᶜ, nxᶠ, nyᶠ, nxᶜ, nyᶜ, cx, cy, px, py)
    I, J, k = @index(Global, NTuple)
    s = zero(eltype(bᶜ))
    @inbounds if Dᶜ[I, J, k] < 0
        for n in 1:4, m in 1:4
            i = fine_index(I, m, nxᶠ, cx, px)
            j = fine_index(J, n, nyᶠ, cy, py)
            (i == 0) | (j == 0) && continue
            Iᵖ, Jᵖ, Iₙ, Jₙ, wx, wy, wxy = interpolation_weights(Dᶜ, i, j, k, nxᶜ, nyᶜ, cx, cy, px, py)
            w = (Iᵖ == I) * (Jᵖ == J) * (1 - wx - wy - wxy) + (Iₙ == I) * (Jᵖ == J) * wx +
                (Iᵖ == I) * (Jₙ == J) * wy                  + (Iₙ == I) * (Jₙ == J) * wxy
            s += w * r[i, j, k]
        end
    end
    @inbounds bᶜ[I, J, k] = s
end

# Thomas factorization of every vertical column: inverse pivots β⁻¹ and multipliers t
@kernel function _factorize_columns!(β⁻¹, t, Cz, D, E, T, Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ε = E[i, j]
        top = T[i, j]
        β = (D[i, j, 1] - ifelse(Nz == 1, top, zero(top))) * (1 + ε)
        β⁻¹[i, j, 1] = 1 / β
        for k in 2:Nz
            Dᵏ = D[i, j, k] - ifelse(k == Nz, top, zero(top))
            tᵏ = Cz[i, j, k] / β
            β = Dᵏ * (1 + ε) - Cz[i, j, k] * tᵏ
            t[i, j, k] = tᵏ
            β⁻¹[i, j, k] = 1 / β
        end
    end
end

# Red-black line relaxation: for each column (i, j) of the given color, solve the vertical
# tridiagonal sub-system exactly with the horizontal couplings moved to the right-hand side
# using the current iterate.
@kernel function _smooth_columns!(ϕ, b, Cx, Cy, Cz, β⁻¹, t, color, Nx, Ny, Nz)
    m, j = @index(Global, NTuple)
    i = 2m - (color + j) % 2
    if i <= Nx
        i⁻ = ifelse(i == 1, Nx, i - 1)
        i⁺ = ifelse(i == Nx, 1, i + 1)
        j⁻ = ifelse(j == 1, Ny, j - 1)
        j⁺ = ifelse(j == Ny, 1, j + 1)

        @inbounds begin
            y = zero(eltype(ϕ))
            for k in 1:Nz
                rhs = b[i, j, k] - (Cx[i, j, k] * ϕ[i⁻, j, k] + Cx[i+1, j, k] * ϕ[i⁺, j, k] +
                                    Cy[i, j, k] * ϕ[i, j⁻, k] + Cy[i, j+1, k] * ϕ[i, j⁺, k])
                y = (rhs - Cz[i, j, k] * y) * β⁻¹[i, j, k]
                ϕ[i, j, k] = y
            end

            for k in Nz-1:-1:1
                y = ϕ[i, j, k] - t[i, j, k+1] * y
                ϕ[i, j, k] = y
            end
        end
    end
end

#####
##### Free-surface (Robin) top-row correction, refreshed whenever Δt changes
#####
##### An implicit free surface turns the rigid-lid Neumann condition at the top into a Robin
##### condition that subtracts Az / (g Δt² + Δzᶠ/2) from the diagonal at k = Nz (matching
##### `FreeSurfaceLaplacian`). Like the vertical conductances, the correction is summed over
##### agglomerated columns.
#####

@kernel function _fine_top_correction!(T, grid, g, Δt, Nz)
    i, j = @index(Global, NTuple)
    Δzᶠ = Δzᵃᵃᶠ(i, j, Nz+1, grid)
    den = g * Δt^2 + Δzᶠ / 2
    @inbounds T[i, j] = Azᶜᶜᶠ(i, j, Nz+1, grid) / den
end

@kernel function _coarsen_top_correction!(Tᶜ, Tᶠ, cx, cy, Nxᶠ, Nyᶠ)
    I, J = @index(Global, NTuple)
    i₁ = ifelse(cx, 2I - 1, I)
    i₂ = ifelse(cx, min(2I, Nxᶠ), I)
    j₁ = ifelse(cy, 2J - 1, J)
    j₂ = ifelse(cy, min(2J, Nyᶠ), J)
    s = zero(eltype(Tᶜ))
    @inbounds for j in j₁:j₂, i in i₁:i₂
        s += Tᶠ[i, j]
    end
    @inbounds Tᶜ[I, J] = s
end

function update_free_surface_correction!(mg::MultigridPreconditioner, free_surface, Δt)
    Δt == mg.cached_free_surface_timestep[] && return nothing
    mg.cached_free_surface_timestep[] = Δt

    grid = mg.grid
    arch = architecture(grid)
    Nz = size(grid)[3]
    g = free_surface.gravitational_acceleration

    fine = first(mg.levels)
    nx, ny, _ = size(fine)
    launch!(arch, grid, (nx, ny), _fine_top_correction!, fine.T, grid, g, Δt, Nz)

    for ℓ in 1:length(mg.levels)-1
        levelᶠ, levelᶜ = mg.levels[ℓ], mg.levels[ℓ+1]
        nxᶠ, nyᶠ, _ = size(levelᶠ)
        nxᶜ, nyᶜ, _ = size(levelᶜ)
        launch!(arch, grid, (nxᶜ, nyᶜ), _coarsen_top_correction!, levelᶜ.T, levelᶠ.T,
                levelᶠ.coarsen_x, levelᶠ.coarsen_y, nxᶠ, nyᶠ)
    end

    factorize_columns!(mg)
    compute_coarse_inverse!(mg)

    return nothing
end

#####
##### Direct solve on the coarsest level with the pseudo-inverse of its operator
#####

function coarsest_operator_matrix(level)
    Cx, Cy, Cz, D, T = (Array{Float64}(on_architecture(CPU(), a)) for a in (level.Cx, level.Cy, level.Cz, level.D, level.T))
    nx, ny, nz = size(D)
    cell(i, j, k) = i + nx * (j - 1 + ny * (k - 1))
    A = zeros(Float64, nx * ny * nz, nx * ny * nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        c = cell(i, j, k)
        i⁻ = ifelse(i == 1, nx, i - 1)
        i⁺ = ifelse(i == nx, 1, i + 1)
        j⁻ = ifelse(j == 1, ny, j - 1)
        j⁺ = ifelse(j == ny, 1, j + 1)
        A[c, c] = D[i, j, k] - ifelse(k == nz, T[i, j], 0.0)
        A[c, cell(i⁻, j, k)] += Cx[i, j, k]
        A[c, cell(i⁺, j, k)] += Cx[i+1, j, k]
        A[c, cell(i, j⁻, k)] += Cy[i, j, k]
        A[c, cell(i, j⁺, k)] += Cy[i, j+1, k]
        k > 1  && (A[c, cell(i, j, k-1)] += Cz[i, j, k])
        k < nz && (A[c, cell(i, j, k+1)] += Cz[i, j, k+1])
    end
    return A
end

# singular values below the cycle's rounding level, the rigid-lid null space among them, are dropped
function compute_coarse_inverse!(mg::MultigridPreconditioner)
    FT = eltype(mg.coarse_inverse)
    A⁺ = pinv(coarsest_operator_matrix(last(mg.levels)); rtol = 100 * eps(FT))
    copyto!(mg.coarse_inverse, Matrix{FT}(A⁺))
    return nothing
end

function factorize_columns!(mg::MultigridPreconditioner)
    grid = mg.grid
    arch = architecture(grid)
    for level in mg.levels
        nx, ny, nz = size(level)
        launch!(arch, grid, (nx, ny), _factorize_columns!, level.β⁻¹, level.t, level.Cz, level.D, level.E, level.T, nz)
    end
    return nothing
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
                          level_array(nx, ny),
                          level_array(nx, ny, nz),
                          level_array(nx, ny, nz),
                          level_array(nx, ny, nz),
                          level_array(nx, ny, nz),
                          level_array(nx, ny, nz),
                          cx, cy)
end

# the coarsest level is solved directly, so it is kept small enough for a dense inverse
const coarsest_level_size = 512

"""
    MultigridPreconditioner(grid; smoothing_sweeps = 2, float_type = eltype(grid))

Construct a geometric multigrid preconditioner for the [`ConjugateGradientPoissonSolver`](@ref)
that approximates `(V∇²)⁻¹` with one V-cycle per application.

The symmetric volume-weighted Laplacian `V∇²` is stored in conductance form (face area ×
inverse center-to-center distance, zeroed across immersed and domain boundaries). Coarse
levels agglomerate cells `2 × 2` in the horizontal, never in the vertical, until the coarsest
level has at most $coarsest_level_size cells, where it is solved directly with the dense
pseudo-inverse of its operator; grid sizes need not be powers of two. Immersed boundaries,
partial cells and stretching in any direction enter every level through the coefficients, so
no FFT-solvability of the grid is required. Transfers between levels are bilinear
interpolation and its transpose. The smoother is red-black line relaxation that solves each
vertical column exactly, applied `smoothing_sweeps` times before and after the coarse-grid
correction (in reversed color order afterwards, so the preconditioner is symmetric). Strong
vertical anisotropy (`Δz ≪ Δx`) is absorbed by the smoother; where the horizontal spacing is
instead much finer than the vertical, convergence degrades.

With an implicit free surface (a [`FreeSurfaceLaplacian`](@ref) linear operation), the Robin
condition's `Δt`-dependent top-row diagonal correction is carried on every level and refreshed
whenever the time step changes.

With `float_type = Float32` the level hierarchy is stored and smoothed in `Float32` while the
conjugate gradient iteration stays in the grid's precision, halving the V-cycle's memory traffic.
This suits stretched rectilinear grids; on strongly anisotropic grids such as deep latitude-longitude
basins the horizontal modes fall below `Float32` resolution and the iteration count grows.

Example
=======

```jldoctest
using Oceananigans
using Oceananigans.Solvers: MultigridPreconditioner

grid = RectilinearGrid(size=(16, 16, 8), extent=(1, 1, 1))
preconditioner = MultigridPreconditioner(grid)

# output
MultigridPreconditioner with 2 levels
├── level 1: 16×16×8
└── level 2: 8×8×8
```
"""
function MultigridPreconditioner(grid::AbstractGrid; smoothing_sweeps = 2, float_type = eltype(grid))
    TX, TY, TZ = topology(grid)
    TZ === Bounded ||
        throw(ArgumentError("MultigridPreconditioner requires a Bounded z-direction (got $TZ)"))

    arch = architecture(grid)
    FT = float_type
    Nx, Ny, Nz = size(grid)

    sizes = [(Nx, Ny)]
    coarsenings = NTuple{2, Bool}[]
    while true
        nx, ny = last(sizes)
        cx = TX !== Flat && nx > 1
        cy = TY !== Flat && ny > 1
        (cx || cy) && nx * ny * Nz > coarsest_level_size || break
        push!(coarsenings, (cx, cy))
        push!(sizes, (cx ? cld(nx, 2) : nx, cy ? cld(ny, 2) : ny))
    end
    push!(coarsenings, (false, false))

    levels = [allocate_multigrid_level(arch, FT, nx, ny, Nz, cx, cy)
              for ((nx, ny), (cx, cy)) in zip(sizes, coarsenings)]

    fine = first(levels)
    TX === Flat ||
        launch!(arch, grid, (Nx + 1, Ny, Nz), _fine_x_conductance!, fine.Cx, grid, TX === Periodic, Nx)
    TY === Flat ||
        launch!(arch, grid, (Nx, Ny + 1, Nz), _fine_y_conductance!, fine.Cy, grid, TY === Periodic, Ny)
    launch!(arch, grid, (Nx, Ny, Nz + 1), _fine_z_conductance!, fine.Cz, grid, Nz)

    for ℓ in 1:length(levels)-1
        levelᶠ, levelᶜ = levels[ℓ], levels[ℓ+1]
        nxᶠ, nyᶠ, _ = size(levelᶠ)
        nxᶜ, nyᶜ, _ = size(levelᶜ)
        cx, cy = levelᶠ.coarsen_x, levelᶠ.coarsen_y
        launch!(arch, grid, (nxᶜ + 1, nyᶜ, Nz), _coarsen_x_conductance!, levelᶜ.Cx, levelᶠ.Cx, cx, cy, nxᶠ, nyᶠ)
        launch!(arch, grid, (nxᶜ, nyᶜ + 1, Nz), _coarsen_y_conductance!, levelᶜ.Cy, levelᶠ.Cy, cx, cy, nxᶠ, nyᶠ)
        launch!(arch, grid, (nxᶜ, nyᶜ, Nz + 1), _coarsen_z_conductance!, levelᶜ.Cz, levelᶠ.Cz, cx, cy, nxᶠ, nyᶠ)
        # a single periodic cell couples only to itself
        nxᶜ == 1 && TX === Periodic && fill!(levelᶜ.Cx, zero(FT))
        nyᶜ == 1 && TY === Periodic && fill!(levelᶜ.Cy, zero(FT))
    end

    ε = convert(FT, 1//100)
    εᶠ = 32 * Nz * eps(FT)
    for level in levels
        nx, ny, nz = size(level)
        launch!(arch, grid, (nx, ny, nz), _compute_diagonal!, level.D, level.Cx, level.Cy, level.Cz)
        launch!(arch, grid, (nx, ny), _compute_column_regularization!, level.E, level.Cx, level.Cy, nz, ε, εᶠ)
    end

    n = prod(size(last(levels)))
    coarse_inverse = on_architecture(arch, zeros(FT, n, n))
    mg = MultigridPreconditioner(grid, levels, coarse_inverse, smoothing_sweeps, Ref(convert(eltype(grid), NaN)))
    factorize_columns!(mg)
    compute_coarse_inverse!(mg)

    return mg
end

#####
##### The V-cycle
#####

function smooth_level!(mg::MultigridPreconditioner, level, colors...)
    grid = mg.grid
    arch = architecture(grid)
    nx, ny, nz = size(level)
    for color in colors
        launch!(arch, grid, (cld(nx, 2), ny),
                _smooth_columns!, level.ϕ, level.b, level.Cx, level.Cy, level.Cz,
                level.β⁻¹, level.t, color, nx, ny, nz)
    end
    return nothing
end

function vcycle!(mg::MultigridPreconditioner, ℓ)
    grid = mg.grid
    arch = architecture(grid)
    level = mg.levels[ℓ]
    nx, ny, nz = size(level)

    if ℓ == length(mg.levels)
        mul!(vec(level.ϕ), mg.coarse_inverse, vec(level.b))
        return nothing
    end

    fill!(level.ϕ, zero(eltype(level.ϕ)))

    for _ in 1:mg.smoothing_sweeps
        smooth_level!(mg, level, 0, 1)
    end

    launch!(arch, grid, (nx, ny, nz), _compute_level_residual!, level.r, level.ϕ, level.b,
            level.Cx, level.Cy, level.Cz, level.D, level.T, nx, ny, nz)

    levelᶜ = mg.levels[ℓ+1]
    nxᶜ, nyᶜ, _ = size(levelᶜ)
    TX, TY, _ = topology(grid)
    px, py = TX === Periodic, TY === Periodic
    cx, cy = level.coarsen_x, level.coarsen_y
    launch!(arch, grid, (nxᶜ, nyᶜ, nz), _restrict_residual!, levelᶜ.b, level.r, levelᶜ.D,
            nx, ny, nxᶜ, nyᶜ, cx, cy, px, py)

    vcycle!(mg, ℓ + 1)

    launch!(arch, grid, (nx, ny, nz), _prolong_and_correct!, level.ϕ, levelᶜ.ϕ, levelᶜ.D, level.D,
            nxᶜ, nyᶜ, cx, cy, px, py)

    for _ in 1:mg.smoothing_sweeps
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
