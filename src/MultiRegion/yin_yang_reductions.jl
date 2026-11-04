using Oceananigans.Fields: convert_mask
using Oceananigans.Operators: Azᶜᶜᵃ, Azᶜᶠᵃ, Azᶠᶜᵃ, Azᶠᶠᵃ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑxyᶠᶠᵃ

# Reductions that combine the panels of a Yin-Yang grid count every point of the sphere once: sums weight every
# point by its owned area fraction w, and the other reductions consider only the points with w > 0. Reductions
# over dims = 3 alone are computed panel by panel over all interior points.

const YinYangOperand = AbstractField{<:Any, <:Any, <:Any, <:YinYangGridOfSomeKind}

function spanning_operand(reduction!, operand::YinYangOperand)
    ℓx, ℓy, _ = instantiated_location(operand)
    return owned_operand(reduction!, operand, yin_yang_masks(operand.grid).cell_weights, ℓx, ℓy)
end

function owned_operand(::SumReduction, operand, weights, ℓx, ℓy)
    LX, LY, LZ = location(operand)
    T = promote_type(eltype(operand), eltype(weights))
    return KernelFunctionOperation{LX, LY, LZ}(owned_weighted_value, operand.grid, (operand, weights, ℓx, ℓy), T)
end

# The points without an owned part take the neutral element of the reduction
function owned_operand(reduction!, operand, weights, ℓx, ℓy)
    LX, LY, LZ = location(operand)
    excluded = convert_mask(eltype(operand), get_neutral_mask(reduction!))
    return KernelFunctionOperation{LX, LY, LZ}(owned_value, operand.grid, (operand, weights, ℓx, ℓy, excluded), eltype(operand))
end

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
Fields.filltype(::SumReduction, c::YinYangOperand) = promote_type(eltype(c), eltype(c.grid))
