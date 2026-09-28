#####
##### Radiation penetrating into the convective layer
#####

# Radiation drives convection only where it is absorbed above the plume, so Deardorff's w★³ = h Jᵇ becomes
# w★³ = W(h) = h Jᵇ + Jʳ ∫₀ʰ T(d) dd, with Jᵇ the total surface buoyancy flux and Jʳ ≥ 0 its radiative part

function transmitted_fraction end
function transmitted_thickness end

@inline transmitted_thickness(::Nothing, i, j, grid, h) = zero(grid)
@inline surface_radiative_buoyancy_flux(i, j, grid, ::Nothing, buoyancy, fields) = zero(grid)

# Buoyancy production `W(h)` integrated over a convective layer of depth `h`, and its derivative `W′(h)`
@inline convective_buoyancy_production(i, j, grid, radiation, h, Jᵇ, Jʳ) = h * Jᵇ + Jʳ * transmitted_thickness(radiation, i, j, grid, h)
@inline convective_buoyancy_production_rate(i, j, grid, radiation, h, Jᵇ, Jʳ) = Jᵇ + Jʳ * transmitted_fraction(radiation, i, j, grid, h)

# Depth `hᶜ` at which `W` is largest, where the radiation absorbed above balances the surface cooling:
# `W′(hᶜ) = Jᵇ + Jʳ T(hᶜ) = 0`. Infinite when the column is cooled on net, `Jᵇ ≥ 0`.
@inline compensation_depth(i, j, grid, ::Nothing, Jᵇ, Jʳ) = convert(eltype(grid), Inf)

# A 30-iteration bisection algorithm to find the compensation depth
@inline function compensation_depth(i, j, grid, radiation, Jᵇ, Jʳ)
    FT = eltype(grid)
    𝒯★ = - Jᵇ / Jʳ
    h⁻ = zero(FT)
    h⁺ = static_column_depthᶜᶜᵃ(i, j, grid)

    for _ in 1:30
        h = (h⁻ + h⁺) / 2
        deeper = transmitted_fraction(radiation, i, j, grid, h) > 𝒯★
        h⁻ = ifelse(deeper, h, h⁻)
        h⁺ = ifelse(deeper, h⁺, h)
    end

    return ifelse(Jᵇ < 0, (h⁻ + h⁺) / 2, FT(Inf))
end

# Convective layer depth `h` solving `w★³ = W(h) + h Jᵇᵋ` by Newton's method, from the `h → 0` limit `W / h → Jᵇ + Jʳ`.
# Below a peak of W at `hᶜ` the steps are taken on `s = √(Wᶜ - W)`, which stays linear in `h` up to the peak, where
# Newton on W itself slows to linear convergence. Above `Wᶜ = W(hᶜ)` there is no root and `h = hᶜ`.
@inline convective_layer_depth(i, j, grid, ::Nothing, w★³, Jᵇ, Jʳ, hᶜ, Jᵇᵋ) = w★³ / (Jᵇ + Jᵇᵋ)

@inline function convective_layer_depth(i, j, grid, radiation, w★³, Jᵇ, Jʳ, hᶜ, Jᵇᵋ)
    peaked = isfinite(hᶜ)
    hᵖ = ifelse(peaked, hᶜ, zero(hᶜ))
    Wᶜ = convective_buoyancy_production(i, j, grid, radiation, hᵖ, Jᵇ, Jʳ) + hᵖ * Jᵇᵋ
    s★ = sqrt(clip(Wᶜ - w★³))

    h = min(hᶜ, w★³ / (Jᵇ + Jʳ + Jᵇᵋ))

    for _ in 1:3
        W  = convective_buoyancy_production(i, j, grid, radiation, h, Jᵇ, Jʳ) + h * Jᵇᵋ
        W′ = convective_buoyancy_production_rate(i, j, grid, radiation, h, Jᵇ, Jʳ) + Jᵇᵋ
        s  = sqrt(clip(Wᶜ - W))
        δh = (w★³ - W) / W′ * ifelse(peaked, 2s / (s + s★), one(s))
        h  = min(hᶜ, h + δh)
    end

    return ifelse(peaked & (w★³ ≥ Wᶜ), hᶜ, h)
end

# Surface buoyancy flux that produces `W(h)` over the convective layer depth `h`.
@inline effective_buoyancy_flux(i, j, grid, ::Nothing, h, Jᵇ, Jʳ) = Jᵇ

@inline function effective_buoyancy_flux(i, j, grid, radiation, h, Jᵇ, Jʳ)
    W = convective_buoyancy_production(i, j, grid, radiation, h, Jᵇ, Jʳ)
    return ifelse(h > 0, W / h, Jᵇ + Jʳ)
end
