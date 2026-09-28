using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: CenterField, Field, set!
using Oceananigans.Utils: launch!, KernelParameters
using Oceananigans.Grids: inactive_node, topology, Periodic, RightCenterFolded, RightFaceFolded
using Oceananigans.Operators: δxᶠᶠᶜ, δyᶠᶠᶜ, δxᶠᶜᶜ, δyᶠᶜᶜ, δxᶜᶠᶜ, δyᶜᶠᶜ, flux_div_xyᶜᶜᶜ, Ax_qᶜᶠᶜ, Ay_qᶠᶜᶜ,
                              Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ, Δzᶠᶠᶜ, Δxᶠᶜᶜ, Δyᶠᶜᶜ, Δxᶜᶠᶜ, Δyᶜᶠᶜ, Δrᵃᵃᶠ

"""
    ShearSignedCoriolis(grid; ε=1/8, η=1/32, saturation=1/2, smoothing=3, update_interval=86400, adjustment_time=30*86400)

Energy-conserving Coriolis scheme plus the increment `grad A + curl* B`, with

    A_c = Σ_z α_cz f_z Γ_z,    B_z = - Σ_c α_cz f_z D_c,

where `Γ` is the area-weighted circulation at the cell corners `z`, `D` the horizontal volume-flux divergence at the
cell centres `c`, and `α_cz` couples each cell to its four corners with the weights `χ ε (a - b) + η (1 - 2 (a - b)²)`
for the corner `(i + a, j + b)`. The increment does no work and has no curl for nondivergent flow for any `α`,
and removes the null modes of the four-point average. The chirality `χ ∈ [-1, 1]` orients the stencil:
`χ = 1` is the south-east stencil, `χ = -1` the north-west one, and `χ(x, y)` is uniform along each column. The chiral part
converts potential energy of grid-scale waves at a rate proportional to `-χ ∫ (∂z u ∂z|p′ₓ|² - ∂z v ∂z|p′ᵧ|²) / N² dz`, where
`p′ₓ` and `p′ᵧ` are the parts of the hydrostatic pressure, free surface included, at wavelengths of a few cells along x and
along y. Every `update_interval`, `χ` relaxes over `adjustment_time` toward the sign of that integral over the stably
stratified interfaces below the uppermost one, smoothed over `smoothing` cells, which makes the conversion negative; `χ = -1`
where there is no grid-scale pressure variance. The couplings use `clamp(χ / saturation, -1, 1)`, so the stencil keeps its
full chirality except where the smoothed `χ` changes sign. Pairs touching land, walls or a tripolar fold are excluded.

Keyword arguments
=================

- `ε`: chiral weight. Default: 1/8.
- `η`: achiral weight. Default: 1/32.
- `saturation`: value of `|χ|` above which the couplings use the full chirality. Default: 1/2.
- `smoothing`: width [cells] of the Gaussian smoothing of the conversion and of its sign. Default: 3.
- `update_interval`: time [s] between updates of the chirality. Default: 1 day.
- `adjustment_time`: time scale [s] of the relaxation of the chirality. Default: 30 days.
"""
struct ShearSignedCoriolis{FT, C, P, S, W, U}
    ε :: FT
    η :: FT
    saturation :: FT
    chirality :: C
    pressure :: P
    streamfunction :: S
    workspace :: W
    smoothing_passes :: Int
    update_interval :: FT
    adjustment_time :: FT
    next_update_time :: U
end

function ShearSignedCoriolis(grid; ε=1/8, η=1/32, saturation=1/2, smoothing=3, update_interval=86400, adjustment_time=30*86400)
    FT = eltype(grid)
    chirality = Field{Center, Center, Nothing}(grid)
    set!(chirality, -1)
    fill_halo_regions!(chirality)
    pressure = CenterField(grid)
    streamfunction = Field{Face, Face, Center}(grid)
    workspace = (; target = CenterField(grid), mask = CenterField(grid),
                  zonal_grid_scale_pressure = CenterField(grid), meridional_grid_scale_pressure = CenterField(grid),
                  column_target = Field{Center, Center, Nothing}(grid), column_buffer = Field{Center, Center, Nothing}(grid),
                  column_mask = Field{Center, Center, Nothing}(grid))
    return ShearSignedCoriolis(FT(ε), FT(η), FT(saturation), chirality, pressure, streamfunction, workspace, round(Int, 2smoothing^2),
                               FT(update_interval), FT(adjustment_time), Ref(zero(FT)))
end

Base.summary(scheme::ShearSignedCoriolis) = "ShearSignedCoriolis(ε=$(scheme.ε), η=$(scheme.η))"

Adapt.adapt_structure(to, scheme::ShearSignedCoriolis) = ShearSignedCoriolis(scheme.ε, scheme.η, scheme.saturation, Adapt.adapt(to, scheme.chirality),
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
    χ = clamp(@inbounds(scheme.chirality[ic, jc, 1]) / scheme.saturation, -1, 1)
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
##### Chirality χ(x, y): relaxed toward the sign of the column-integrated chiral conversion Σ (∂z u Δ|p′ₓ|² - ∂z v Δ|p′ᵧ|²) / N²,
##### with p′ₓ, p′ᵧ the grid-scale parts of the hydrostatic pressure along x and y, smoothed; χ = -1 without grid-scale variance
#####

# u and v averaged over the wet faces of the cell, and whether both averages have a wet face
@inline function cell_velocity(i, j, k, grid, u, v)
    wu⁻ = !peripheral_node(i,   j, k, grid, face, center, center)
    wu⁺ = !peripheral_node(i+1, j, k, grid, face, center, center)
    wv⁻ = !peripheral_node(i, j,   k, grid, center, face, center)
    wv⁺ = !peripheral_node(i, j+1, k, grid, center, face, center)
    ū = @inbounds (wu⁻ * u[i, j, k] + wu⁺ * u[i+1, j, k]) / max(wu⁻ + wu⁺, 1)
    v̄ = @inbounds (wv⁻ * v[i, j, k] + wv⁺ * v[i, j+1, k]) / max(wv⁻ + wv⁺, 1)
    return ū, v̄, (wu⁻ | wu⁺) & (wv⁻ | wv⁺)
end

# ∂z u Δ|p′ₓ|² - ∂z v Δ|p′ᵧ|² across the interface between levels k and k + 1; the uppermost interface lies in the surface
# Ekman layer
@inline function interface_conversion(i, j, k, grid, u, v, p′ₓ, p′ᵧ)
    ū⁻, v̄⁻, lower_defined = cell_velocity(i, j, k,   grid, u, v)
    ū⁺, v̄⁺, upper_defined = cell_velocity(i, j, k+1, grid, u, v)
    defined = (1 ≤ k ≤ size(grid, 3) - 2) & lower_defined & upper_defined &
              !inactive_node(i, j, k, grid, center, center, center) & !inactive_node(i, j, k+1, grid, center, center, center)
    zonal = @inbounds (ū⁺ - ū⁻) * (p′ₓ[i, j, k+1]^2 - p′ₓ[i, j, k]^2)
    meridional = @inbounds (v̄⁺ - v̄⁻) * (p′ᵧ[i, j, k+1]^2 - p′ᵧ[i, j, k]^2)
    return (zonal - meridional) / Δrᵃᵃᶠ(i, j, k+1, grid), defined
end

# b = ∂z pHY′ on the face below the center k
@inline function face_buoyancy(i, j, k, grid, p)
    Δz = znode(i, j, k, grid, center, center, center) - znode(i, j, k-1, grid, center, center, center)
    return @inbounds (p[i, j, k] - p[i, j, k-1]) / Δz
end

# N² on the interface between levels k and k + 1, from b on the nearest faces that lie in the water
@inline function interface_stratification(i, j, k, grid, p)
    Nz = size(grid, 3)
    below = (k ≥ 2) & !inactive_node(i, j, k-1, grid, center, center, center)
    above = (k + 2 ≤ Nz) & !inactive_node(i, j, k+2, grid, center, center, center)
    k⁻ = ifelse(below, k, k + 1)
    k⁺ = ifelse(above, k + 2, k + 1)
    Δb = face_buoyancy(i, j, k⁺, grid, p) - face_buoyancy(i, j, k⁻, grid, p)
    Δz = znode(i, j, k⁺, grid, center, center, face) - znode(i, j, k⁻, grid, center, center, face)
    return Δb / Δz, k⁻ < k⁺
end

@kernel function _compute_column_conversion!(conversion, defined, grid, u, v, pHY′, p′ₓ, p′ᵧ)
    i, j, _ = @index(Global, NTuple)
    C = zero(grid)
    n = 0
    for k in 1:size(grid, 3) - 2
        density, interface_defined = interface_conversion(i, j, k, grid, u, v, p′ₓ, p′ᵧ)
        N², resolved = interface_stratification(i, j, k, grid, pHY′)
        stable = interface_defined & resolved & (N² > 0)
        C += ifelse(stable, density / N², zero(grid))
        n += stable
    end
    @inbounds conversion[i, j, 1] = C
    @inbounds defined[i, j, 1] = n > 0
end

@kernel function _wet_mask!(mask, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds mask[i, j, k] = !inactive_node(i, j, k, grid, center, center, center)
end

@kernel function _total_pressure!(p, pHY′, η, g, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds p[i, j, k] = pHY′[i, j, k] + g * η[i, j, grid.Nz+1]
end

@kernel function _subtract_from!(p′, p, mask)
    i, j, k = @index(Global, NTuple)
    @inbounds p′[i, j, k] = (p[i, j, k] - p′[i, j, k]) * mask[i, j, k]
end

@kernel function _sign_of_conversion!(target, mask, grid)
    i, j, _ = @index(Global, NTuple)
    @inbounds target[i, j, 1] = ifelse((mask[i, j, 1] > 0) & (target[i, j, 1] > 0), 1, -1)
    @inbounds mask[i, j, 1] = !inactive_node(i, j, size(grid, 3), grid, center, center, center)
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

function masked_smooth!(ψ, buffer, mask, passes, workspec)
    grid = ψ.grid
    arch = architecture(grid)
    for _ in 1:passes
        launch!(arch, grid, workspec, _smooth_along_x!, buffer, ψ, mask, grid)
        launch!(arch, grid, workspec, _smooth_along_y!, ψ, buffer, mask, grid)
    end
    return nothing
end

@kernel function _relax_chirality!(χ, target, rate)
    i, j, _ = @index(Global, NTuple)
    @inbounds χ[i, j, 1] += rate * (target[i, j, 1] - χ[i, j, 1])
end

function update_chirality!(scheme::ShearSignedCoriolis, velocities, pressure_anomaly, displacement, g)
    (; target, mask) = scheme.workspace
    p′ₓ, p′ᵧ = scheme.workspace.zonal_grid_scale_pressure, scheme.workspace.meridional_grid_scale_pressure
    grid = target.grid
    arch = architecture(grid)

    # p = pHY′ + g η minus one masked (1, 2, 1) pass along x (p′ₓ) or along y (p′ᵧ): a quarter of the second difference of p
    launch!(arch, grid, :xyz, _wet_mask!, mask, grid)
    launch!(arch, grid, :xyz, _total_pressure!, target, pressure_anomaly, displacement, g, grid)
    launch!(arch, grid, :xyz, _smooth_along_x!, p′ₓ, target, mask, grid)
    launch!(arch, grid, :xyz, _subtract_from!, p′ₓ, target, mask)
    launch!(arch, grid, :xyz, _smooth_along_y!, p′ᵧ, target, mask, grid)
    launch!(arch, grid, :xyz, _subtract_from!, p′ᵧ, target, mask)

    (; column_target, column_buffer, column_mask) = scheme.workspace
    columns = KernelParameters(1:size(grid, 1), 1:size(grid, 2), 1:1)
    launch!(arch, grid, columns, _compute_column_conversion!, column_target, column_mask, grid, velocities.u, velocities.v, pressure_anomaly, p′ₓ, p′ᵧ)
    masked_smooth!(column_target, column_buffer, column_mask, scheme.smoothing_passes, columns)
    launch!(arch, grid, columns, _sign_of_conversion!, column_target, column_mask, grid)
    masked_smooth!(column_target, column_buffer, column_mask, scheme.smoothing_passes, columns)
    launch!(arch, grid, columns, _relax_chirality!, scheme.chirality, column_target, scheme.update_interval / scheme.adjustment_time)
    fill_halo_regions!(scheme.chirality)
    return nothing
end

update_coriolis!(coriolis, model) = nothing

function update_coriolis!(coriolis::SSC, model)
    scheme = coriolis.scheme
    grid = model.grid
    u, v = model.velocities.u, model.velocities.v

    if model.clock.time ≥ scheme.next_update_time[]
        free_surface = model.free_surface
        update_chirality!(scheme, model.velocities, model.pressure.pHY′, free_surface.displacement, free_surface.gravitational_acceleration)
        scheme.next_update_time[] = (fld(model.clock.time, scheme.update_interval) + 1) * scheme.update_interval
    end

    launch!(architecture(grid), grid, :xyz, _compute_coriolis_potentials!, scheme.pressure, scheme.streamfunction, grid, coriolis, u, v)
    fill_halo_regions!((scheme.pressure, scheme.streamfunction))
    return nothing
end
