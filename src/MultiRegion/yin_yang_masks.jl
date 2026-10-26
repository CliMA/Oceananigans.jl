# Ownership of the points of a Yin-Yang panel, and the owned area fractions of its cells: a panel owns a point
# when the point lies deeper inside its 270° × 90° core than inside the other panel's. In the first quadrant the
# dividing curve φ★(λ) between the owned regions has three closed-form pieces P₁, P₂, P₃, which meet at the corners
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
##### Exact owned area of a cell
#####

# The latitude φ★(λ) ≥ 0 of the dividing curve above the longitude 0 ≤ λ ≤ 3π/4 (radians): the owned points of
# the column λ are exactly |φ| < φ★.
function dividing_curve_latitude(λ)
    if λ ≤ dividing_curve_corner_longitude
        return atan(sqrt(cos(λ)))                     # P₁: tan²φ★ = cos λ
    elseif λ ≤ π - dividing_curve_corner_longitude
        return atan(sin(λ))                           # P₂: tan φ★ = sin λ
    else
        return acos(clamp(-cos(λ) / sin(λ), -1, 1))   # P₃: cos φ★ = -cot λ
    end
end

# S(λ) = ∫₀^λ sin φ★ dλ′ for 0 ≤ λ ≤ 3π/4: on P₁ with t = tan(λ/2), on P₂ directly, and on P₃ parametrized
# by φ★ with λ★(φ) = π/2 + atan(cos φ), where sin φ★ dλ = -sin²φ / (1 + cos²φ) dφ.
function integrated_dividing_curve_sine(λ)
    λ₁, λ₂ = dividing_curve_corner_longitude, π - dividing_curve_corner_longitude
    S₁(λ) = (t = tan(λ / 2); 2 * atan(√2 * t / √(1 - t^2)) - √2 * asin(t))
    S₂(λ) = asin(cos(λ₁) / √2) - asin(cos(λ) / √2)
    G(φ) = √2 * atan(tan(φ) / √2) - φ
    S₃(λ) = G(dividing_curve_corner_latitude) - G(dividing_curve_latitude(λ))

    λ ≤ λ₁ && return S₁(λ)
    λ ≤ λ₂ && return S₁(λ₁) + S₂(λ)
    return S₁(λ₁) + S₂(λ₂) + S₃(λ)
end

# The owned area (radius 1) of the rectangle [0, λ] × [0, φ] in the first quadrant, ∫₀^λ sin(min(φ★, φ)) dλ′:
# the integrand is sin φ on the owned longitudes [0, b₁] ∪ [a₂, b₂] of the row φ and sin φ★ elsewhere.
function first_quadrant_owned_area(λ, φ)
    λ = min(λ, core_half_width)
    φ = min(φ, core_half_height)
    if φ < dividing_curve_corner_latitude
        b₁ = π/2 + atan(cos(φ))      # the row crosses P₃ only
        a₂ = b₂ = b₁
    else
        b₁ = acos(min(tan(φ)^2, 1)) # the row crosses P₁, then P₂ twice
        a₂ = asin(min(tan(φ), 1))
        b₂ = π - a₂
    end

    S = integrated_dividing_curve_sine
    function owned(a, b)
        a, b = min(a, λ), min(b, λ)
        return sin(φ) * (b - a) - (S(b) - S(a))
    end
    return S(λ) + owned(0, b₁) + owned(a₂, b₂)
end

# The owned fraction of the cell [λ₁, λ₂] × [φ₁, φ₂] (radians, any signs). The owned region is symmetric about
# λ = 0 and φ = 0, so the signed area `A` of [0, λ] × [0, φ] is a cumulative function of the owned area measure.
function owned_cell_fraction(λ₁, λ₂, φ₁, φ₂)
    A(λ, φ) = sign(λ) * sign(φ) * first_quadrant_owned_area(abs(λ), abs(φ))
    column_area(λ) = A(λ, φ₂) - A(λ, φ₁)
    area = column_area(λ₂) - column_area(λ₁)
    return clamp(area / ((λ₂ - λ₁) * (sin(φ₂) - sin(φ₁))), 0, 1)
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

# The start of the first candidate stencil that lies within the `Nx × Ny` interior, or `nothing`
function interior_stencil_start(i★, j★, s, (Nx, Ny))
    for j₀ in stencil_starts(j★, s), i₀ in stencil_starts(i★, s)
        (1 ≤ i₀ && i₀ + s - 1 ≤ Nx && 1 ≤ j₀ && j₀ + s - 1 ≤ Ny) && return (i₀, j₀)
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

# The owned weights of a panel with cell faces at `(λᶠ, φᶠ)` (degrees) and horizontal halo `(Hx, Hy)`, in the
# halo-inclusive layout of a `Field{Center, Center, Nothing}` and shared by both panels. The owned cells of the
# two panels tile the sphere: the weighted areas of each panel sum to 2π on the unit sphere, to rounding error.
function YinYangMasks(arch, FT, (λᶠ, φᶠ), (Hx, Hy))
    λʳ, φʳ = deg2rad.(Float64.(λᶠ)), deg2rad.(Float64.(φᶠ))
    Nx, Ny = length(λᶠ) - 1, length(φᶠ) - 1
    cell_weights = OffsetArrays.OffsetArray(zeros(FT, Nx + 2Hx, Ny + 2Hy, 1), -Hx, -Hy, 0)
    for j in 1:Ny, i in 1:Nx
        cell_weights[i, j, 1] = owned_cell_fraction(λʳ[i], λʳ[i+1], φʳ[j], φʳ[j+1])
    end
    return YinYangMasks(on_architecture(arch, cell_weights))
end
