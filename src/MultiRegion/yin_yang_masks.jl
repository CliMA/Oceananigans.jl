# Ownership of the points of a Yin-Yang panel, and the owned area fractions of its cells: a panel owns a point
# when the point lies deeper inside its 270° × 90° core than inside the other panel's. In the first quadrant the
# dividing curve between the owned regions has three closed-form pieces, which meet at the corners
# (acos g, asin g) and (π - acos g, asin g) with g = (√5 - 1) / 2.

const golden_ratio_conjugate = (√5 - 1) / 2
const dividing_curve_corner_latitude = asin(golden_ratio_conjugate)
const dividing_curve_corner_longitude = acos(golden_ratio_conjugate)
const core_half_width = 3π / 4
const core_half_height = π / 4

# The coordinates (degrees) in the frame of one panel of the point at `(λ, φ)` in the frame of the other:
# the map (x, y, z) ↦ (-x, z, y) is an involution, so the same function serves both directions.
@inline function yin_yang_transform(λ, φ)
    λʳ, φʳ = deg2rad(λ), deg2rad(φ)
    x′ = - cos(φʳ) * cos(λʳ)
    y′ = sin(φʳ)
    z′ = cos(φʳ) * sin(λʳ)
    return rad2deg(atan(y′, x′)), rad2deg(atan(z′, sqrt(x′^2 + y′^2)))
end

# Whether the panel in whose frame `(λ, φ)` (degrees) are given owns the point
yin_yang_margin(λ, φ) = min(45 - abs(φ), 135 - abs(λ))

function yin_yang_owned(λ, φ)
    λ″, φ″ = yin_yang_transform(λ, φ)
    return yin_yang_margin(λ, φ) > yin_yang_margin(λ″, φ″)
end

#####
##### The dividing curve as a graph φ★(|λ|) and its inverse
#####

# Latitude φ★(λ) ≥ 0 of the dividing curve above longitude λ (radians): the owned points of
# the column |λ| are exactly |φ| < φ★. Zero for |λ| ≥ 3π/4.
function dividing_curve_latitude(λ)
    λ = abs(λ)
    if λ ≤ dividing_curve_corner_longitude
        return atan(sqrt(cos(λ)))
    elseif λ ≤ π - dividing_curve_corner_longitude
        return atan(sin(λ))
    elseif λ < core_half_width
        return acos(clamp(-cos(λ) / sin(λ), -1, 1))
    else
        return zero(λ)
    end
end

# φ★ is piecewise monotone, with a local minimum and a kink at the corners and a maximum at λ = π/2.
const dividing_curve_critical_longitudes = (dividing_curve_corner_longitude, π/2, π - dividing_curve_corner_longitude)

# The owned longitudes of the row |φ| (radians), restricted to λ ≥ 0, as two intervals
# [0, b₁] ∪ [a₂, b₂] (the second is empty, a₂ = b₂, for |φ| below the corner latitude).
function dividing_curve_owned_longitudes(φ)
    φ = abs(φ)
    if φ ≥ core_half_height
        return (zero(φ), zero(φ), zero(φ), zero(φ))
    elseif φ < dividing_curve_corner_latitude
        b₁ = π/2 + atan(cos(φ))
        return (zero(φ), b₁, b₁, b₁)
    else
        t = tan(φ)
        b₁ = acos(clamp(t^2, -1, 1))
        a₂ = asin(clamp(t, -1, 1))
        return (zero(φ), b₁, a₂, π - a₂)
    end
end

#####
##### Exact owned area of a cell
#####

# ∫ₐᵇ sin φ★(λ) dλ for [a, b] within a single piece of the dividing curve.
function integrate_dividing_curve_sine(a, b)
    λ = (a + b) / 2
    if λ ≤ dividing_curve_corner_longitude
        # P₁: sin φ★ = √(cos λ / (1 + cos λ)), and with tan(λ/2) = sin θ, sin φ★ dλ = √2 cos²θ dθ / (1 + sin²θ)
        F(λ) = (t = tan(λ / 2); 2 * atan(√2 * t / √(1 - t^2)) - √2 * asin(t))
        return F(b) - F(a)
    elseif λ ≤ π - dividing_curve_corner_longitude
        # P₂: sin φ★ = sin λ / √(1 + sin²λ) = d/dλ [-asin(cos λ / √2)]
        return asin(cos(a) / √2) - asin(cos(b) / √2)
    else
        # P₃ parametrized by φ: λ★(φ) = π/2 + atan(cos φ), ∫ sin φ (-λ★′) dφ = ∫ sin²φ / (1 + cos²φ) dφ
        G(φ) = √2 * atan(tan(φ) / √2) - φ
        return G(dividing_curve_latitude(a)) - G(dividing_curve_latitude(b))
    end
end

# Owned area (radius 1) of the rectangle [λ₁, λ₂] × [φ₁, φ₂] in the first quadrant,
# 0 ≤ λ₁ ≤ λ₂ ≤ 3π/4 and 0 ≤ φ₁ ≤ φ₂ ≤ π/4, as ∫ [sin(clip(φ★(λ), φ₁, φ₂)) - sin φ₁] dλ,
# split where the clipped curve has kinks: at the critical longitudes and where φ★ crosses φ₁ or φ₂.
function first_quadrant_owned_area(λ₁, λ₂, φ₁, φ₂)
    crossings = (dividing_curve_owned_longitudes(φ₁)..., dividing_curve_owned_longitudes(φ₂)...)
    kinks = filter(λ -> λ₁ < λ < λ₂, (dividing_curve_critical_longitudes..., crossings...))
    breakpoints = sort!([λ₁, λ₂, kinks...])

    area = zero(λ₁)
    for (a, b) in zip(breakpoints[1:end-1], breakpoints[2:end])
        φ★ = dividing_curve_latitude((a + b) / 2)
        if φ★ ≥ φ₂
            area += (b - a) * (sin(φ₂) - sin(φ₁))
        elseif φ★ > φ₁
            area += integrate_dividing_curve_sine(a, b) - (b - a) * sin(φ₁)
        end
    end
    return area
end

# Fraction of the rectangle [λ₁, λ₂] × [φ₁, φ₂] (radians, any signs) that is owned.
# Exactly 1 for cells entirely under the dividing curve and exactly 0 for cells entirely
# above it; cut cells are split at λ = 0 and φ = 0 and reflected into the first quadrant.
function owned_cell_fraction(λ₁, λ₂, φ₁, φ₂)
    λlow = λ₁ ≤ 0 ≤ λ₂ ? zero(λ₁) : min(abs(λ₁), abs(λ₂))
    λhigh = max(abs(λ₁), abs(λ₂))
    φlow = φ₁ ≤ 0 ≤ φ₂ ? zero(φ₁) : min(abs(φ₁), abs(φ₂))
    φhigh = max(abs(φ₁), abs(φ₂))

    (λlow ≥ core_half_width || φlow ≥ core_half_height) && return zero(λ₁)

    # φ★ is piecewise monotone, so its extrema over the cell lie at its edges or at the critical longitudes inside
    λs = (λlow, min(λhigh, core_half_width), filter(λ -> λlow < λ < λhigh, dividing_curve_critical_longitudes)...)
    φ★ = dividing_curve_latitude.(λs)
    (λhigh ≤ core_half_width && minimum(φ★) ≥ φhigh) && return one(λ₁)
    maximum(φ★) ≤ φlow && return zero(λ₁)

    area = zero(λ₁)
    for (a, b) in ((max(λ₁, 0), max(λ₂, 0)), (max(-λ₂, 0), max(-λ₁, 0))),
        (c, d) in ((max(φ₁, 0), max(φ₂, 0)), (max(-φ₂, 0), max(-φ₁, 0)))

        area += first_quadrant_owned_area(min(a, core_half_width), min(b, core_half_width),
                                          min(c, core_half_height), min(d, core_half_height))
    end

    return area / ((λ₂ - λ₁) * (sin(φ₂) - sin(φ₁)))
end

# The owned area fractions of the cells of a panel with cell faces at `(λᶠ, φᶠ)` (degrees), which serve both panels.
# Their areas sum to 2π on the unit sphere to rounding error.
function yin_yang_cell_weights(λᶠ, φᶠ)
    λʳ = deg2rad.(Float64.(λᶠ))
    φʳ = deg2rad.(Float64.(φᶠ))
    return [owned_cell_fraction(λʳ[i], λʳ[i+1], φʳ[j], φʳ[j+1]) for i in 1:length(λᶠ)-1, j in 1:length(φᶠ)-1]
end

#####
##### Halo nodes and their donor stencils: each panel fills its halo nodes by interpolation from the interior
##### nodes of the other panel, at the receiver's own location and, for vector components, at the other component's
#####

# Index offset of a location along a panel axis: node `i` at `Face` lies on the cell face
# λᶠ[i], and at `Center` half a cell beyond it.
@inline location_index_offset(::Face) = 0
@inline location_index_offset(::Center) = 1//2

# Panel-frame spacing and first-face coordinates `(; Δλ, Δφ, λ₀, φ₀)` in Float64.
panel_frame(λᶠ, φᶠ) = (; Δλ = Float64(λᶠ[2] - λᶠ[1]), Δφ = Float64(φᶠ[2] - φᶠ[1]),
                         λ₀ = Float64(λᶠ[1]), φ₀ = Float64(φᶠ[1]))

# The fractional indices `(i★, j★)` of the point with panel coordinates `(λ, φ)` in a panel's
# index space at `(ℓx, ℓy)`.
function panel_fractional_indices(λ, φ, frame, (ℓx, ℓy))
    i★ = (λ - frame.λ₀) / frame.Δλ + 1 - location_index_offset(ℓx)
    j★ = (φ - frame.φ₀) / frame.Δφ + 1 - location_index_offset(ℓy)
    return i★, j★
end

# The fractional indices `(i★, j★)` in the other panel's index space at `donor_location` of
# the node `(i, j)` at `(ℓx, ℓy)`, and the node's coordinates `(λ″, φ″)` in the other panel's frame.
function donor_fractional_indices(i, j, frame, (ℓx, ℓy), donor_location)
    λ′ = frame.λ₀ + (i - 1 + location_index_offset(ℓx)) * frame.Δλ
    φ′ = frame.φ₀ + (j - 1 + location_index_offset(ℓy)) * frame.Δφ
    λ″, φ″ = yin_yang_transform(λ′, φ′)
    return panel_fractional_indices(λ″, φ″, frame, donor_location)..., λ″, φ″
end

stencil_start(x★, s) = floor(Int, x★) - s ÷ 2 + 1

# Starting indices of the s-wide stencils that bracket the fractional index x★. A point within
# `tolerance` of an integer (receiver nodes map exactly onto donor nodes where the panels touch)
# has the stencils on both sides as candidates.
stencil_starts(x★, s; tolerance = 1e-6) = unique((stencil_start(x★ - tolerance, s), stencil_start(x★ + tolerance, s)))

function interior_stencil_start(i★, j★, s, (Nx, Ny))
    for j₀ in stencil_starts(j★, s), i₀ in stencil_starts(i★, s)
        west_edge_inside = i₀ ≥ 1
        east_edge_inside = i₀ + s - 1 ≤ Nx
        south_edge_inside = j₀ ≥ 1
        north_edge_inside = j₀ + s - 1 ≤ Ny
        stencil_inside = west_edge_inside & east_edge_inside & south_edge_inside & north_edge_inside
        stencil_inside && return (i₀, j₀)
    end
    return nothing
end

# The halo nodes of a panel with `Nx × Ny` interior cells and horizontal halo `(Hx, Hy)`.
halo_nodes((Nx, Ny), (Hx, Hy)) = [(i, j) for j in 1-Hy:Ny+Hy for i in 1-Hx:Nx+Hx if !(1 ≤ i ≤ Nx && 1 ≤ j ≤ Ny)]

#####
##### Owned weights stored on a YinYangGrid
#####

struct YinYangMasks{W}
    cell_weights :: W # owned area fractions of the (Center, Center) cells, zero in the halos
end

Grids.on_architecture(arch, masks::YinYangMasks) = YinYangMasks(on_architecture(arch, masks.cell_weights))

# The owned weights of a panel with cell faces at `(λᶠ, φᶠ)` and horizontal halo `(Hx, Hy)`, in the
# halo-inclusive layout of a `Field{Center, Center, Nothing}` and shared by both panels.
function YinYangMasks(arch, FT, (λᶠ, φᶠ), (Hx, Hy))
    Nx, Ny = length(λᶠ) - 1, length(φᶠ) - 1
    cell_weights = OffsetArray(zeros(FT, Nx + 2Hx, Ny + 2Hy, 1), -Hx, -Hy, 0)
    view(cell_weights, 1:Nx, 1:Ny, 1) .= yin_yang_cell_weights(λᶠ, φᶠ)
    return YinYangMasks(on_architecture(arch, cell_weights))
end
