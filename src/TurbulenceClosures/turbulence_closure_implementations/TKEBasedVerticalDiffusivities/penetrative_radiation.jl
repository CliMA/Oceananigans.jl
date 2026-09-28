#####
##### Radiation penetrating into the convective layer
#####

# A layer of depth h, kept well mixed by convection, loses buoyancy through the surface at the total rate Jᵇ, the
# radiation included, while a part Jʳ T(h) of the radiative flux Jʳ ≥ 0 leaves through its base without driving
# the turbulence. Deardorff's w★³ = h Jᵇ becomes w★³ = W(h) = h (Jᵇ - Jʳ T(h)) + 2 Jʳ ∫₀ʰ T(d) dd.

function transmitted_fraction end
function transmitted_fraction_derivative end
function transmitted_thickness end

@inline surface_radiative_buoyancy_flux(i, j, grid, ::Nothing, buoyancy, fields) = zero(grid)

# Buoyancy production `W(h)` over a convective layer of depth `h`, and its derivative `W′(h)`
@inline convective_buoyancy_production(i, j, grid, ::Nothing, h, Jᵇ, Jʳ) = h * Jᵇ

@inline function convective_buoyancy_production(i, j, grid, radiation, h, Jᵇ, Jʳ)
    𝒯 = transmitted_fraction(radiation, i, j, grid, h)
    ℒ = transmitted_thickness(radiation, i, j, grid, h)
    return h * (Jᵇ - Jʳ * 𝒯) + 2 * Jʳ * ℒ
end

@inline function convective_buoyancy_production_rate(i, j, grid, radiation, h, Jᵇ, Jʳ)
    𝒯  = transmitted_fraction(radiation, i, j, grid, h)
    𝒯′ = transmitted_fraction_derivative(radiation, i, j, grid, h)
    return Jᵇ + Jʳ * (𝒯 - h * 𝒯′)
end

# Convective layer depth `h` solving `w★³ = W(h) + h Jᵇᵋ` in a convecting column, `Jᵇ > 0`, where `W` is concave and
# increasing. Newton's method converges monotonically from below, here from the larger root of the two lines `W`
# lies under: the surface tangent `h (Jᵇ + Jʳ)` and the deep asymptote `h Jᵇ + 2 Jʳ ∫₀ᴴ T`.
@inline convective_layer_depth(i, j, grid, ::Nothing, w★³, Jᵇ, Jʳ, Jᵇᵋ) = w★³ / (Jᵇ + Jᵇᵋ)

@inline function convective_layer_depth(i, j, grid, radiation, w★³, Jᵇ, Jʳ, Jᵇᵋ)
    H = static_column_depthᶜᶜᵃ(i, j, grid)
    ℒ = transmitted_thickness(radiation, i, j, grid, H)
    h = max(w★³ / (Jᵇ + Jʳ + Jᵇᵋ), (w★³ - 2 * Jʳ * ℒ) / (Jᵇ + Jᵇᵋ))

    for _ in 1:3
        W  = convective_buoyancy_production(i, j, grid, radiation, h, Jᵇ, Jʳ) + h * Jᵇᵋ
        W′ = convective_buoyancy_production_rate(i, j, grid, radiation, h, Jᵇ, Jʳ) + Jᵇᵋ
        h  = h + (w★³ - W) / W′
    end

    return h
end

# Surface buoyancy flux that produces `W(h)` over the convective layer depth `h`.
@inline effective_buoyancy_flux(i, j, grid, ::Nothing, h, Jᵇ, Jʳ) = Jᵇ

@inline function effective_buoyancy_flux(i, j, grid, radiation, h, Jᵇ, Jʳ)
    W = convective_buoyancy_production(i, j, grid, radiation, h, Jᵇ, Jʳ)
    return ifelse(h > 0, W / h, Jᵇ + Jʳ)
end
