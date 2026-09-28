using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: CenterField, Field, set!
using Oceananigans.Utils: launch!
using Oceananigans.Grids: inactive_node, topology, Periodic, RightCenterFolded, RightFaceFolded
using Oceananigans.Operators: δxᶠᶠᶜ, δyᶠᶠᶜ, δxᶠᶜᶜ, δyᶠᶜᶜ, δxᶜᶠᶜ, δyᶜᶠᶜ, flux_div_xyᶜᶜᶜ, Ax_qᶜᶠᶜ, Ay_qᶠᶜᶜ,
                              Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ, Δzᶠᶠᶜ, Δxᶠᶜᶜ, Δyᶠᶜᶜ, Δxᶜᶠᶜ, Δyᶜᶠᶜ, Δrᵃᵃᶠ

"""
    ShearSignedCoriolis(grid; ε=1/8, η=1/32, smoothing=3, update_interval=86400, adjustment_time=30*86400)

Energy-conserving Coriolis scheme plus the increment `grad A + curl* B`, with

    A_c = Σ_z α_cz f_z Γ_z,    B_z = - Σ_c α_cz f_z D_c,

where `Γ` is the area-weighted circulation at the cell corners `z`, `D` the horizontal volume-flux divergence at the
cell centres `c`, and `α_cz` couples each cell to its four corners with the weights `χ ε (a - b) + η (1 - 2 (a - b)²)`
for the corner `(i + a, j + b)`. The increment does no work and has no curl for nondivergent flow for any `α`,
and removes the null modes of the four-point average. The chirality `χ ∈ [-1, 1]` orients the stencil:
`χ = 1` is the south-east stencil, `χ = -1` the north-west one. Every `update_interval` it relaxes, over
`adjustment_time`, toward the sign of the vertical shear of `u - v` below the uppermost interface, smoothed over
`smoothing` cells; `χ = -1` without shear. Pairs touching land, walls or a tripolar fold are excluded.

Keyword arguments
=================

- `ε`: chiral weight. Default: 1/8.
- `η`: achiral weight. Default: 1/32.
- `smoothing`: width [cells] of the Gaussian smoothing of the shear and of its sign. Default: 3.
- `update_interval`: time [s] between updates of the chirality. Default: 1 day.
- `adjustment_time`: time scale [s] of the relaxation of the chirality. Default: 30 days.
"""
struct ShearSignedCoriolis{FT, C, P, S, W, U}
    ε :: FT
    η :: FT
    chirality :: C
    pressure :: P
    streamfunction :: S
    workspace :: W
    smoothing_passes :: Int
    update_interval :: FT
    adjustment_time :: FT
    next_update_time :: U
end

function ShearSignedCoriolis(grid; ε=1/8, η=1/32, smoothing=3, update_interval=86400, adjustment_time=30*86400)
    FT = eltype(grid)
    chirality = CenterField(grid)
    set!(chirality, -1)
    fill_halo_regions!(chirality)
    pressure = CenterField(grid)
    streamfunction = Field{Face, Face, Center}(grid)
    workspace = (; target = CenterField(grid), buffer = CenterField(grid), mask = CenterField(grid))
    return ShearSignedCoriolis(FT(ε), FT(η), chirality, pressure, streamfunction, workspace, round(Int, 2smoothing^2),
                               FT(update_interval), FT(adjustment_time), Ref(zero(FT)))
end

Base.summary(scheme::ShearSignedCoriolis) = "ShearSignedCoriolis(ε=$(scheme.ε), η=$(scheme.η))"

Adapt.adapt_structure(to, scheme::ShearSignedCoriolis) = ShearSignedCoriolis(scheme.ε, scheme.η, Adapt.adapt(to, scheme.chirality),
                                                                             Adapt.adapt(to, scheme.pressure), Adapt.adapt(to, scheme.streamfunction),
                                                                             nothing, scheme.smoothing_passes, scheme.update_interval,
                                                                             scheme.adjustment_time, nothing)

const SSC = AbstractRotation{<:ShearSignedCoriolis}

Oceananigans.prognostic_state(coriolis::SSC) = (; χ = Oceananigans.prognostic_state(coriolis.scheme.chirality),
                                                  next_update_time = coriolis.scheme.next_update_time[])

function Oceananigans.restore_prognostic_state!(coriolis::SSC, from::NamedTuple)
    Oceananigans.restore_prognostic_state!(coriolis.scheme.chirality, from.χ)
    coriolis.scheme.next_update_time[] = from.next_update_time
    return coriolis
end

#####
##### Pair couplings and the potentials A (cell centres) and B (cell corners)
#####

@inline folded_row(j, grid) = (topology(grid, 2) <: Union{RightCenterFolded, RightFaceFolded}) & (j ≥ size(grid, 2))

# Corners on the domain walls are peripheral nodes, so this also drops the pairs that reach beyond a wall
@inline coastal_corner(i, j, k, grid) = peripheral_node(i, j, k, grid, face, face, center) | inactive_node(i, j, k, grid, face, face, center)

@inline circulationᶠᶠᶜ(i, j, k, grid, u, v) = δxᶠᶠᶜ(i, j, k, grid, Ax_qᶜᶠᶜ, v) - δyᶠᶠᶜ(i, j, k, grid, Ay_qᶠᶜᶜ, u)

# α f for the cell (ic, jc) and its corner (ic + a, jc + b); the aspect-ratio factor caps the response of modes
# varying along the short side of a cell at ≈ f
@inline function pair_coupling(ic, jc, a, b, k, grid, coriolis)
    iz, jz = ic + a, jc + b
    scheme = coriolis.scheme
    χ = @inbounds scheme.chirality[ic, jc, k]
    w = χ * scheme.ε * (a - b) + scheme.η * (1 - 2 * (a - b)^2)
    excluded = folded_row(jc, grid) | folded_row(jz, grid) | coastal_corner(iz, jz, k, grid) | inactive_node(ic, jc, k, grid, center, center, center)
    Δx = Δxᶜᶜᶜ(ic, jc, k, grid)
    Δy = Δyᶜᶜᶜ(ic, jc, k, grid)
    α = w * min(Δx, Δy) / max(Δx, Δy) / sqrt(Δzᶜᶜᶜ(ic, jc, k, grid) * Δzᶠᶠᶜ(iz, jz, k, grid))
    return ifelse(excluded, zero(grid), α * fᶠᶠᵃ(iz, jz, k, grid, coriolis))
end

@inline function coriolis_pressureᶜᶜᶜ(i, j, k, grid, coriolis, u, v)
    A = zero(grid)
    for a in 0:1, b in 0:1
        A += pair_coupling(i, j, a, b, k, grid, coriolis) * circulationᶠᶠᶜ(i+a, j+b, k, grid, u, v)
    end
    return A
end

@inline function coriolis_streamfunctionᶠᶠᶜ(i, j, k, grid, coriolis, u, v)
    B = zero(grid)
    for a in 0:1, b in 0:1
        B -= pair_coupling(i-a, j-b, a, b, k, grid, coriolis) * flux_div_xyᶜᶜᶜ(i-a, j-b, k, grid, u, v)
    end
    return B
end

@kernel function _compute_coriolis_potentials!(A, B, grid, coriolis, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds A[i, j, k] = coriolis_pressureᶜᶜᶜ(i, j, k, grid, coriolis, u, v)
    @inbounds B[i, j, k] = coriolis_streamfunctionᶠᶠᶜ(i, j, k, grid, coriolis, u, v)
end

# Plain differences along the layer are the adjoints of D and Γ that make the increment skew
@inline function x_f_cross_U(i, j, k, grid, coriolis::SSC, U)
    A, B = coriolis.scheme.pressure, coriolis.scheme.streamfunction
    increment = δxᶠᶜᶜ(i, j, k, grid, A) / Δxᶠᶜᶜ(i, j, k, grid) - δyᶠᶜᶜ(i, j, k, grid, B) / Δyᶠᶜᶜ(i, j, k, grid)
    return - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_Ay_vᶠᶠᶜ, coriolis, U[2]) * Ay⁻¹ᶠᶜᶜ(i, j, k, grid) - increment
end

@inline function y_f_cross_U(i, j, k, grid, coriolis::SSC, U)
    A, B = coriolis.scheme.pressure, coriolis.scheme.streamfunction
    increment = δyᶜᶠᶜ(i, j, k, grid, A) / Δyᶜᶠᶜ(i, j, k, grid) + δxᶜᶠᶜ(i, j, k, grid, B) / Δxᶜᶠᶜ(i, j, k, grid)
    return + ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_Ax_uᶠᶠᶜ, coriolis, U[1]) * Ax⁻¹ᶜᶠᶜ(i, j, k, grid) - increment
end

#####
##### Chirality: relaxed toward the smoothed sign of the smoothed shear of u - v
#####

# u - v averaged over the wet faces of the cell, and whether both averages have a wet face
@inline function projected_velocity(i, j, k, grid, u, v)
    wu⁻ = !peripheral_node(i,   j, k, grid, face, center, center)
    wu⁺ = !peripheral_node(i+1, j, k, grid, face, center, center)
    wv⁻ = !peripheral_node(i, j,   k, grid, center, face, center)
    wv⁺ = !peripheral_node(i, j+1, k, grid, center, face, center)
    ū = @inbounds (wu⁻ * u[i, j, k] + wu⁺ * u[i+1, j, k]) / max(wu⁻ + wu⁺, 1)
    v̄ = @inbounds (wv⁻ * v[i, j, k] + wv⁺ * v[i, j+1, k]) / max(wv⁻ + wv⁺, 1)
    return ū - v̄, (wu⁻ | wu⁺) & (wv⁻ | wv⁺)
end

# Shear across the interface between levels k and k + 1; the uppermost interface lies in the surface Ekman layer
@inline function interface_shear(i, j, k, grid, u, v)
    lower, lower_defined = projected_velocity(i, j, k,   grid, u, v)
    upper, upper_defined = projected_velocity(i, j, k+1, grid, u, v)
    defined = (1 ≤ k ≤ size(grid, 3) - 2) & lower_defined & upper_defined &
              !inactive_node(i, j, k, grid, center, center, center) & !inactive_node(i, j, k+1, grid, center, center, center)
    return (upper - lower) / Δrᵃᵃᶠ(i, j, k+1, grid), defined
end

@kernel function _compute_level_shear!(shear, sheared, grid, u, v)
    i, j, k = @index(Global, NTuple)
    Nz = size(grid, 3)
    k⁻ = ifelse(k == Nz, Nz - 2, k - 1)
    k⁺ = ifelse(k == Nz, Nz - 2, k)
    lower_shear, lower_defined = interface_shear(i, j, k⁻, grid, u, v)
    upper_shear, upper_defined = interface_shear(i, j, k⁺, grid, u, v)
    n = lower_defined + upper_defined
    @inbounds shear[i, j, k] = (lower_defined * lower_shear + upper_defined * upper_shear) / max(n, 1)
    @inbounds sheared[i, j, k] = !inactive_node(i, j, k, grid, center, center, center) & (n > 0)
end

@kernel function _sign_of_shear!(target, mask, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds target[i, j, k] = ifelse((mask[i, j, k] > 0) & (target[i, j, k] > 0), 1, -1)
    @inbounds mask[i, j, k] = !inactive_node(i, j, k, grid, center, center, center)
end

# Masked (1, 2, 1) passes; x is periodic when the grid is, walls otherwise
@kernel function _smooth_along_x!(ψx, ψ, mask, grid)
    i, j, k = @index(Global, NTuple)
    Nx = size(grid, 1)
    periodic = topology(grid, 1) === Periodic
    i⁻ = ifelse(periodic, mod1(i - 1, Nx), max(i - 1, 1))
    i⁺ = ifelse(periodic, mod1(i + 1, Nx), min(i + 1, Nx))
    @inbounds begin
        m⁻ = ifelse(periodic | (i > 1),  mask[i⁻, j, k], zero(eltype(mask)))
        m⁺ = ifelse(periodic | (i < Nx), mask[i⁺, j, k], zero(eltype(mask)))
        m = mask[i, j, k]
        s = m⁻ + 2m + m⁺
        ψx[i, j, k] = ifelse(s > 0, (m⁻ * ψ[i⁻, j, k] + 2m * ψ[i, j, k] + m⁺ * ψ[i⁺, j, k]) / s, zero(eltype(ψ)))
    end
end

@kernel function _smooth_along_y!(ψ, ψx, mask, grid)
    i, j, k = @index(Global, NTuple)
    Ny = size(grid, 2)
    j⁻ = max(j - 1, 1)
    j⁺ = min(j + 1, Ny)
    @inbounds begin
        m⁻ = ifelse(j > 1,  mask[i, j⁻, k], zero(eltype(mask)))
        m⁺ = ifelse(j < Ny, mask[i, j⁺, k], zero(eltype(mask)))
        m = mask[i, j, k]
        s = m⁻ + 2m + m⁺
        ψ[i, j, k] = m * ifelse(s > 0, (m⁻ * ψx[i, j⁻, k] + 2m * ψx[i, j, k] + m⁺ * ψx[i, j⁺, k]) / s, zero(eltype(ψ)))
    end
end

function masked_smooth!(ψ, buffer, mask, passes)
    grid = ψ.grid
    arch = architecture(grid)
    for _ in 1:passes
        launch!(arch, grid, :xyz, _smooth_along_x!, buffer, ψ, mask, grid)
        launch!(arch, grid, :xyz, _smooth_along_y!, ψ, buffer, mask, grid)
    end
    return nothing
end

@kernel function _relax_chirality!(χ, target, rate)
    i, j, k = @index(Global, NTuple)
    @inbounds χ[i, j, k] += rate * (target[i, j, k] - χ[i, j, k])
end

function update_chirality!(scheme::ShearSignedCoriolis, velocities)
    (; target, buffer, mask) = scheme.workspace
    grid = target.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _compute_level_shear!, target, mask, grid, velocities.u, velocities.v)
    masked_smooth!(target, buffer, mask, scheme.smoothing_passes)
    launch!(arch, grid, :xyz, _sign_of_shear!, target, mask, grid)
    masked_smooth!(target, buffer, mask, scheme.smoothing_passes)
    launch!(arch, grid, :xyz, _relax_chirality!, scheme.chirality, target, scheme.update_interval / scheme.adjustment_time)
    fill_halo_regions!(scheme.chirality)
    return nothing
end

update_coriolis!(coriolis, model) = nothing

function update_coriolis!(coriolis::SSC, model)
    scheme = coriolis.scheme
    grid = model.grid
    u, v = model.velocities.u, model.velocities.v

    if model.clock.time ≥ scheme.next_update_time[]
        update_chirality!(scheme, model.velocities)
        scheme.next_update_time[] = (fld(model.clock.time, scheme.update_interval) + 1) * scheme.update_interval
    end

    launch!(architecture(grid), grid, :xyz, _compute_coriolis_potentials!, scheme.pressure, scheme.streamfunction, grid, coriolis, u, v)
    fill_halo_regions!((scheme.pressure, scheme.streamfunction))
    return nothing
end
