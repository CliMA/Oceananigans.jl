using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ, ĝ_z
using Oceananigans.Grids: c, f, rnode, inactive_node
using Oceananigans.ImmersedBoundaries: BottomAndTopIBG

# The topmost wet cell is integrated from the immersed top
update_hydrostatic_pressure!(pHY′, arch, ibg::BottomAndTopIBG, buoyancy, tracers; parameters = surface_kernel_parameters(ibg)) =
    launch!(arch, ibg, parameters, _update_hydrostatic_pressure!, pHY′, ibg, buoyancy, tracers)

update_hydrostatic_pressure!(::Nothing, arch, ::BottomAndTopIBG, args...; kw...) = nothing

@inline function upper_half_spacing(i, j, k, grid)
    zᶜ = rnode(i, j, k, grid, c, c, c)
    return ifelse(k == grid.Nz,
                  rnode(i, j, k + 1, grid, c, c, f) - zᶜ,
                  (rnode(i, j, k + 1, grid, c, c, c) - zᶜ) / 2)
end

@inline function lower_half_spacing(i, j, k, grid)
    zᶜ = rnode(i, j, k, grid, c, c, c)
    return ifelse(k == 1,
                  zᶜ - rnode(i, j, k, grid, c, c, f),
                  (zᶜ - rnode(i, j, k - 1, grid, c, c, c)) / 2)
end

@inline function topmost_active_index(i, j, grid)
    kᵗ = 0
    for k in grid.Nz : -1 : 1
        active = !inactive_node(i, j, k, grid, c, c, c)
        kᵗ = ifelse((kᵗ == 0) & active, k, kᵗ)
    end
    return kᵗ
end

@kernel function _update_hydrostatic_pressure!(pHY′, ibg::BottomAndTopIBG, buoyancy, C)
    i, j = @index(Global, NTuple)
    kᵗ = topmost_active_index(i, j, ibg)
    rᵗ = @inbounds ibg.immersed_boundary.top_height[i, j, 1]
    Φ = column_top_load(i, j, ibg, ibg.immersed_boundary.top_load)
    integrate_bottom_and_top_hydrostatic_pressure!(pHY′, i, j, ibg, kᵗ, rᵗ, Φ, buoyancy, C)
end

@inline column_top_load(i, j, grid, ::Nothing) = zero(grid)
@inline column_top_load(i, j, grid, top_load) = @inbounds top_load[i, j, 1]

@inline integrate_bottom_and_top_hydrostatic_pressure!(pHY′, i, j, ibg, kᵗ, rᵗ, Φ, ::Nothing, C) = nothing

# The top load is added to every level of the column
@inline function integrate_bottom_and_top_hydrostatic_pressure!(pHY′, i, j, ibg, kᵗ, rᵗ, Φ, buoyancy, C)
    grid = ibg.underlying_grid
    # ∂p/∂z = b, so the pressure anomaly increases downward where b < 0
    ĝ = - ĝ_z(buoyancy)
    pᶠ = zero(ĝ)

    for k in ibg.Nz : -1 : 1
        # No contribution from the overlying solid, whose density is unknown
        bᵏ = ifelse(k > kᵗ, zero(ĝ), buoyancy_perturbationᶜᶜᶜ(i, j, k, ibg, buoyancy.formulation, C))

        zᶜ = rnode(i, j, k,     grid, c, c, c)
        z⁺ = rnode(i, j, k + 1, grid, c, c, f)
        z⁻ = rnode(i, j, k,     grid, c, c, f)
        Δr⁺ = upper_half_spacing(i, j, k, grid)
        Δr⁻ = lower_half_spacing(i, j, k, grid)

        δr = rᵗ - zᶜ
        pᵗ = ĝ * bᵏ * (max(zero(δr), δr) * Δr⁺ / (z⁺ - zᶜ) +
                       min(zero(δr), δr) * Δr⁻ / (zᶜ - z⁻))
        pⁱ = pᶠ + ĝ * bᵏ * Δr⁺

        pᶜ = ifelse(k == kᵗ, pᵗ, pⁱ)
        @inbounds pHY′[i, j, k] = pᶜ + Φ
        pᶠ = pᶜ + ĝ * bᵏ * Δr⁻
    end

    return nothing
end
