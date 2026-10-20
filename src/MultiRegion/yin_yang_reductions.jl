using Oceananigans.Operators: Azᶜᶜᵃ, Azᶜᶠᵃ, Azᶠᶜᵃ, Azᶠᶠᵃ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑxyᶠᶠᵃ

# Reductions that combine the panels of a Yin-Yang grid count every point of the sphere once: sums weight every
# point by its owned area fraction w, and the other reductions consider only the points with w > 0. Reductions
# over dims = 3 alone are computed panel by panel over all interior points.

struct YinYangRestriction{R, W}
    reduction! :: R
    weights :: W # owned area fractions of the (Center, Center) cells
end

spanning_restriction(reduction!, grid::YinYangGridOfSomeKind) =
    YinYangRestriction(reduction!, yin_yang_masks(grid).cell_weights)

function restrict_operand(operand, restriction::YinYangRestriction, mask)
    ℓx, ℓy, _ = instantiated_location(operand)
    return restrict_operand(operand, restriction.reduction!, restriction.weights, mask, ℓx, ℓy)
end

# Every region holds the same values of a horizontally reduced operand
restrict_operand(operand, reduction!, weights, mask, ℓx, ℓy) = operand

const HorizontalLocation = Union{Center, Face}

function restrict_operand(operand, ::SumReduction, weights, mask, ℓx::HorizontalLocation, ℓy::HorizontalLocation)
    LX, LY, LZ = location(operand)
    T = promote_type(eltype(operand), eltype(weights))
    return KernelFunctionOperation{LX, LY, LZ}(owned_weighted_value, operand.grid, (operand, weights, ℓx, ℓy), T)
end

function restrict_operand(operand, reduction!, weights, mask, ℓx::HorizontalLocation, ℓy::HorizontalLocation)
    LX, LY, LZ = location(operand)
    excluded = excluded_value(reduction!, mask)
    T = promote_type(eltype(operand), typeof(excluded))
    return KernelFunctionOperation{LX, LY, LZ}(owned_value, operand.grid, (operand, weights, ℓx, ℓy, excluded), T)
end

# The value of the points without an owned part: the reduction's mask, or the neutral element of `any` and `all`
excluded_value(reduction!, mask) = mask
excluded_value(::AnyReduction, mask) = false
excluded_value(::AllReduction, mask) = true

# The owned fraction at (ℓx, ℓy), lumped from the area-weighted fractions of the adjacent (Center, Center) cells
@inline owned_area(i, j, k, grid, w) = @inbounds w[i, j, 1] * Azᶜᶜᵃ(i, j, k, grid)

@inline owned_weight(i, j, k, grid, w, ::Center, ::Center) = @inbounds w[i, j, 1]
@inline owned_weight(i, j, k, grid, w, ::Face,   ::Center) = ℑxᶠᵃᵃ(i, j, k, grid, owned_area, w) / Azᶠᶜᵃ(i, j, k, grid)
@inline owned_weight(i, j, k, grid, w, ::Center, ::Face)   = ℑyᵃᶠᵃ(i, j, k, grid, owned_area, w) / Azᶜᶠᵃ(i, j, k, grid)
@inline owned_weight(i, j, k, grid, w, ::Face,   ::Face)   = ℑxyᶠᶠᵃ(i, j, k, grid, owned_area, w) / Azᶠᶠᵃ(i, j, k, grid)

# Points without an owned part contribute zero, even NaNs
@inline function owned_weighted_value(i, j, k, grid, operand, weights, ℓx, ℓy)
    w = owned_weight(i, j, k, grid, weights, ℓx, ℓy)
    weighted = w * @inbounds(operand[i, j, k])
    return ifelse(w > 0, weighted, zero(weighted))
end

@inline function owned_value(i, j, k, grid, operand, weights, ℓx, ℓy, excluded)
    owned = owned_weight(i, j, k, grid, weights, ℓx, ℓy) > 0
    return ifelse(owned, @inbounds(operand[i, j, k]), excluded)
end

# Weighted sums are fractional, even for integer or Boolean operands
Fields.filltype(::SumReduction, c::AbstractField{<:Any, <:Any, <:Any, <:YinYangGridOfSomeKind}) =
    promote_type(eltype(c), eltype(c.grid))

# Means over both panels divide by the owned number of points, Σ w
function multi_region_length(c, grid::YinYangGrid, dims)
    spans_partition(grid, dims) || return conditional_length(getregion(c, 1), dims)
    return sum(conditional_one(ConditionalOperation(c), 0); dims)
end
