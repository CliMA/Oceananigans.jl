#####
##### Ownership of the points of a Yin-Yang panel, and the owned area fractions of its cells
#####
##### A panel owns a point when the point lies deeper inside its 270° × 90° core than inside the other panel's.
##### In the first quadrant the dividing curve between the owned regions has three closed-form pieces,
##### which meet at the corners (acos g, asin g) and (π - acos g, asin g) with g = (√5 - 1) / 2.
#####

const golden_ratio_conjugate = (√5 - 1) / 2
const dividing_curve_corner_latitude = asin(golden_ratio_conjugate)
const dividing_curve_corner_longitude = acos(golden_ratio_conjugate)
const core_half_width = 3π / 4
const core_half_height = π / 4

# The coordinates (degrees) in the frame of one panel of the point at `(λ, φ)` in the frame of the other:
# the map (x, y, z) ↦ (-x, z, y) is an involution, so the same function serves both directions.
@inline function yin_yang_transform(λ, φ)
    λ′, φ′ = yin_yang_transform_radians(deg2rad(λ), deg2rad(φ))
    return rad2deg(λ′), rad2deg(φ′)
end

@inline function yin_yang_transform_radians(λ, φ)
    x′ = - cos(φ) * cos(λ)
    y′ = sin(φ)
    z′ = cos(φ) * sin(λ)

    λ′ = atan(y′, x′)
    φ′ = atan(z′, sqrt(x′^2 + y′^2))

    return λ′, φ′
end

@inline yin_yang_margin_radians(λ, φ) = min(core_half_height - abs(φ), core_half_width - abs(λ))

# Whether the panel in whose frame `(λ, φ)` (degrees) are given owns the point
function yin_yang_owned(λ, φ)
    λʳ = deg2rad(λ)
    φʳ = deg2rad(φ)
    λ″, φ″ = yin_yang_transform_radians(λʳ, φʳ)
    return yin_yang_margin_radians(λʳ, φʳ) > yin_yang_margin_radians(λ″, φ″)
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

function dividing_curve_latitude_extrema(λ₁, λ₂)
    φmin = min(dividing_curve_latitude(λ₁), dividing_curve_latitude(λ₂))
    φmax = max(dividing_curve_latitude(λ₁), dividing_curve_latitude(λ₂))
    for λ in dividing_curve_critical_longitudes
        if λ₁ < λ < λ₂
            φ = dividing_curve_latitude(λ)
            φmin = min(φmin, φ)
            φmax = max(φmax, φ)
        end
    end
    return φmin, φmax
end

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

# Longitudes λ ∈ [0, 3π/4] at which φ★(λ) = φ, padded to three values.
function dividing_curve_crossings(φ)
    _, b₁, a₂, b₂ = dividing_curve_owned_longitudes(φ)
    return (b₁, a₂, b₂)
end

#####
##### Exact owned area of a cell
#####

# Nodes and weights of the n-point Gauss-Legendre rule on [-1, 1] by Newton iteration on Pₙ.
function gauss_legendre_rule(n)
    nodes = zeros(n)
    weights = zeros(n)
    for i in 1:(n + 1) ÷ 2
        z = cos(π * (i - 1/4) / (n + 1/2))
        derivative = zero(z)
        for _ in 1:100
            p₁, p₂ = one(z), zero(z)
            for j in 1:n
                p₁, p₂ = ((2j - 1) * z * p₁ - (j - 1) * p₂) / j, p₁
            end
            derivative = n * (z * p₁ - p₂) / (z^2 - 1)
            δz = p₁ / derivative
            z -= δz
            abs(δz) < 1e-16 && break
        end
        nodes[i] = -z
        nodes[n + 1 - i] = z
        weights[i] = 2 / ((1 - z^2) * derivative^2)
        weights[n + 1 - i] = weights[i]
    end
    return nodes, weights
end

# ∫ₐᵇ sin φ★(λ) dλ for [a, b] within a single piece of the dividing curve.
function integrate_dividing_curve_sine(a, b, rule)
    λ = (a + b) / 2
    if λ ≤ dividing_curve_corner_longitude
        # P₁: sin φ★ = √(cos λ / (1 + cos λ)) is an elliptic integrand; Gauss-Legendre
        nodes, weights = rule
        h = (b - a) / 2
        integral = zero(λ)
        for (x, w) in zip(nodes, weights)
            integral += w * sin(dividing_curve_latitude(λ + h * x))
        end
        return h * integral
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
# 0 ≤ λ₁ ≤ λ₂ ≤ 3π/4 and 0 ≤ φ₁ ≤ φ₂ ≤ π/4, as ∫ [sin(clip(φ★(λ), φ₁, φ₂)) - sin φ₁] dλ.
# The integrand is split at the kinks of the clipped curve, where each piece is analytic.
function first_quadrant_owned_area(λ₁, λ₂, φ₁, φ₂, rule)
    breakpoints = [λ₁, λ₂]
    for λ in dividing_curve_critical_longitudes
        λ₁ < λ < λ₂ && push!(breakpoints, λ)
    end
    for φ in (φ₁, φ₂), λ in dividing_curve_crossings(φ)
        λ₁ < λ < λ₂ && push!(breakpoints, λ)
    end
    sort!(breakpoints)

    area = zero(λ₁)
    for n in 1:length(breakpoints)-1
        a, b = breakpoints[n], breakpoints[n+1]
        b > a || continue
        φ★ = dividing_curve_latitude((a + b) / 2)
        if φ★ ≥ φ₂
            area += (b - a) * (sin(φ₂) - sin(φ₁))
        elseif φ★ > φ₁
            area += integrate_dividing_curve_sine(a, b, rule) - (b - a) * sin(φ₁)
        end
    end
    return area
end

# Fraction of the rectangle [λ₁, λ₂] × [φ₁, φ₂] (radians, any signs) that is owned.
# Exactly 1 for cells entirely under the dividing curve and exactly 0 for cells entirely
# above it; cut cells are split at λ = 0 and φ = 0 and reflected into the first quadrant.
function owned_cell_fraction(λ₁, λ₂, φ₁, φ₂, rule)
    λlow = λ₁ ≤ 0 ≤ λ₂ ? zero(λ₁) : min(abs(λ₁), abs(λ₂))
    λhigh = max(abs(λ₁), abs(λ₂))
    φlow = φ₁ ≤ 0 ≤ φ₂ ? zero(φ₁) : min(abs(φ₁), abs(φ₂))
    φhigh = max(abs(φ₁), abs(φ₂))

    (λlow ≥ core_half_width || φlow ≥ core_half_height) && return zero(λ₁)
    φmin, φmax = dividing_curve_latitude_extrema(λlow, min(λhigh, core_half_width))
    (λhigh ≤ core_half_width && φmin ≥ φhigh) && return one(λ₁)
    φmax ≤ φlow && return zero(λ₁)

    area = zero(λ₁)
    for (a, b) in ((max(λ₁, 0), max(λ₂, 0)), (max(-λ₂, 0), max(-λ₁, 0)))
        a, b = min(a, core_half_width), min(b, core_half_width)
        for (c, d) in ((max(φ₁, 0), max(φ₂, 0)), (max(-φ₂, 0), max(-φ₁, 0)))
            c, d = min(c, core_half_height), min(d, core_half_height)
            if b > a && d > c
                area += first_quadrant_owned_area(a, b, c, d, rule)
            end
        end
    end

    return area / ((λ₂ - λ₁) * (sin(φ₂) - sin(φ₁)))
end

# The owned area fractions of the cells of a panel with cell faces at `(λᶠ, φᶠ)` (degrees), which serve both panels.
# Their areas sum to 2π on the unit sphere to rounding error.
function yin_yang_cell_weights(λᶠ, φᶠ)
    rule = gauss_legendre_rule(32)
    λʳ = deg2rad.(Float64.(λᶠ))
    φʳ = deg2rad.(Float64.(φᶠ))
    return [owned_cell_fraction(λʳ[i], λʳ[i+1], φʳ[j], φʳ[j+1], rule) for i in 1:length(λᶠ)-1, j in 1:length(φᶠ)-1]
end

#####
##### Halo nodes and their donor stencils
#####
##### Each panel integrates its interior cells and fills its halo nodes by interpolation from the interior
##### nodes of the other panel, at the receiver's own location for scalars and, for the components of a
##### vector, also at the location of the other component.
#####

# Index offset of a location along a panel axis: node `i` at `Face` lies on the cell face
# λᶠ[i], and at `Center` half a cell beyond it.
@inline location_index_offset(::Face) = 0
@inline location_index_offset(::Center) = 1//2

# Panel-frame spacing and first-face coordinates `(; Δλ, Δφ, λ₀, φ₀)` in Float64.
function panel_frame(λᶠ, φᶠ)
    Δλ = Float64(λᶠ[2] - λᶠ[1])
    Δφ = Float64(φᶠ[2] - φᶠ[1])
    λ₀ = Float64(λᶠ[1])
    φ₀ = Float64(φᶠ[1])
    return (; Δλ, Δφ, λ₀, φ₀)
end

# The fractional indices `(i★, j★)` in the other panel's index space at `donor_location` of
# the node `(i, j)` at `receiver_location`, and the node's coordinates `(λ″, φ″)` in the other
# panel's frame.
function donor_fractional_indices(i, j, frame, receiver_location, donor_location)
    ℓx, ℓy = receiver_location
    λ′ = frame.λ₀ + (i - 1 + location_index_offset(ℓx)) * frame.Δλ
    φ′ = frame.φ₀ + (j - 1 + location_index_offset(ℓy)) * frame.Δφ
    λ″, φ″ = yin_yang_transform(λ′, φ′)
    i★, j★ = panel_fractional_indices(λ″, φ″, frame, donor_location)
    return i★, j★, λ″, φ″
end

# The fractional indices `(i★, j★)` of the point with panel coordinates `(λ, φ)` in a panel's
# index space at `location`.
function panel_fractional_indices(λ, φ, frame, location)
    ℓx, ℓy = location
    i★ = (λ - frame.λ₀) / frame.Δλ + 1 - location_index_offset(ℓx)
    j★ = (φ - frame.φ₀) / frame.Δφ + 1 - location_index_offset(ℓy)
    return i★, j★
end

# Starting indices of the s-wide stencils that bracket the fractional index x★. A point within
# `tolerance` of an integer (receiver nodes map exactly onto donor nodes where the panels touch)
# has the stencils on both sides as candidates.
function stencil_starts(x★, s; tolerance = 1e-6)
    lower = stencil_start(x★ - tolerance, s)
    upper = stencil_start(x★ + tolerance, s)
    return lower == upper ? (lower,) : (lower, upper)
end

stencil_start(x★, s) = floor(Int, x★) - s ÷ 2 + 1

# The start `(i₀, j₀)` of an `s × s` stencil of the interior nodes `1:Nx × 1:Ny` of a panel around
# `(i★, j★)`, or `nothing` if there is none.
function interior_stencil_start(i★, j★, s, (Nx, Ny))
    for j₀ in stencil_starts(j★, s), i₀ in stencil_starts(i★, s)
        1 ≤ i₀ && i₀ + s - 1 ≤ Nx && 1 ≤ j₀ && j₀ + s - 1 ≤ Ny && return (i₀, j₀)
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
    Nx = length(λᶠ) - 1
    Ny = length(φᶠ) - 1
    cell_weights = OffsetArray(zeros(FT, Nx + 2Hx, Ny + 2Hy, 1), -Hx, -Hy, 0)
    view(cell_weights, 1:Nx, 1:Ny, 1) .= yin_yang_cell_weights(λᶠ, φᶠ)
    return YinYangMasks(on_architecture(arch, cell_weights))
end
