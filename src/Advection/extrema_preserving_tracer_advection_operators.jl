using Oceananigans.Grids: inactive_node, topology, Flat
using Oceananigans.Operators: V⁻¹ᶜᶜᶜ
using Oceananigans.AbstractOperations: KernelFunctionOperation

# one max.f64 or min.f64 on GPUs, where max and min also propagate NaN and order signed zeros; div_fast multiplies by the reciprocal,
# as the WENO weights do
using Base.FastMath: max_fast, min_fast, div_fast

#####
##### The extrema, set at the first state update
#####

@inline function half_slope(i, j, k, grid, dir, c)
    i⁻, j⁻, k⁻ = stencil_node(i, j, k, dir, -1)
    i⁺, j⁺, k⁺ = stencil_node(i, j, k, dir, +1)
    cᵢ = @inbounds c[i, j, k]
    Δ⁻ = cᵢ - @inbounds c[i⁻, j⁻, k⁻]
    Δ⁺ = @inbounds(c[i⁺, j⁺, k⁺]) - cᵢ
    l  = !inactive_node(i⁻, j⁻, k⁻, grid, Center(), Center(), Center())
    r  = !inactive_node(i⁺, j⁺, k⁺, grid, Center(), Center(), Center())
    minmod = ifelse(Δ⁻ * Δ⁺ > 0, min_fast(abs(Δ⁻), abs(Δ⁺)), zero(cᵢ))
    slope  = ifelse(l & r, minmod, ifelse(l, abs(Δ⁻), ifelse(r, abs(Δ⁺), zero(cᵢ))))
    return slope / 2
end

@inline half_slope(i, j, k, grid::XFG, ::Val{1}, c) = zero(grid)
@inline half_slope(i, j, k, grid::YFG, ::Val{2}, c) = zero(grid)
@inline half_slope(i, j, k, grid::ZFG, ::Val{3}, c) = zero(grid)

# the cell value widened by half its largest minmod slope, below for `s = -1` and above for `s = +1`; one-sided next to a cell without data
@inline function point_value(i, j, k, grid, c, s)
    σ = max_fast(half_slope(i, j, k, grid, Val(1), c), 
                 half_slope(i, j, k, grid, Val(2), c), 
                 half_slope(i, j, k, grid, Val(3), c))
    return @inbounds c[i, j, k] + s * σ
end

"""
$(TYPEDSIGNATURES)

Refresh the time step of `scheme` and the extrema of the tracer `c` it advects. Before the model steps (`Δτ = ∞`) the extrema are cleared,
and the first update of a run sets them to the range of the point values of `c`.
"""
function update_bounds_preserving_limiter!(scheme::ExtremaPreservingWENO, model, c)
    isnothing(c) && return nothing
    grid = model.grid
    E = scheme.bounds.extrema
    scheme.bounds.Δτ[] = adaptive_advection_timestep(model.timestepper, model.clock)

    if isinf(scheme.bounds.Δτ[])
        fill!(E, (convert(eltype(grid), Inf), convert(eltype(grid), -Inf)))
        scheme.bounds.initialized[] = false
    elseif !scheme.bounds.initialized[]
        # every lower point value lies below the upper one of its cell, so the two passes keep the lowest lower and highest upper value
        extrema!(E, KernelFunctionOperation{Center, Center, Center}(point_value, grid, c, -1); init = false)
        extrema!(E, KernelFunctionOperation{Center, Center, Center}(point_value, grid, c, +1); init = false)
        scheme.bounds.initialized[] = true
    end

    return nothing
end

#####
##### The face fluxes, limited in one pass
#####

# selects on the sign bit, an integer compare; GPUs without a Float64 max compare on the FP64 pipe
@inline positive_part(x) = ifelse(signbit(x), zero(x), x)

# θ = min(1, Q / P), the fraction of the antidiffusive flux `A` that keeps the cell's update along one direction within (m, M),
# widened by the cell, what flows into it and the height (ε⁻, ε⁺) of a smooth extremum; a face without flow, as at a wall, adds nothing
@inline function antidiffusive_fraction(cᶜ, κ, A, (qᶠ, cᶠ), (qᵒ, cᵒ), lowers, α, (m, M), (ε⁻, ε⁺))
    cᵘ = cᶜ + κ * (positive_part(qᶠ) * (cᶠ - cᶜ) + positive_part(qᵒ) * (cᵒ - cᶜ))
    Aᵒ = α * abs(qᵒ) * positive_part(ifelse(lowers, cᵒ - cᶜ, cᶜ - cᵒ))
    cᶠ⁺ = ifelse(qᶠ > 0, cᶠ, cᶜ)
    cᵒ⁺ = ifelse(qᵒ > 0, cᵒ, cᶜ)
    lower = min_fast(m, min_fast(cᶜ, cᶠ⁺, cᵒ⁺) - ε⁻)
    upper = max_fast(M, max_fast(cᶜ, cᶠ⁺, cᵒ⁺) + ε⁺)
    Q  = positive_part(ifelse(lowers, cᵘ - lower, upper - cᵘ))
    P  = κ * (abs(A) + Aᵒ)
    # Q never exceeds the denominator, which keeps the quotient off the GPU slow path
    return div_fast(Q, max_fast(P, Q, floatmin(P)))
end

# q (c_D + θ (ψᶠ - c_D)), with θ the smaller antidiffusive fraction of the two cells of the face
@inline function reconstructed_flux(i, j, k, grid, dir, ℓ, advection::ExtremaPreservingWENO, td, u, c, ψᶠ)
    i⁺,  j⁺,  k⁺  = stencil_node(i, j, k, dir, +1)
    i⁻,  j⁻,  k⁻  = stencil_node(i, j, k, dir, -1)
    i⁻⁻, j⁻⁻, k⁻⁻ = stencil_node(i, j, k, dir, -2)
    q  = explicit_transport(i,  j,  k,  grid, dir, ℓ, advection, td, u)
    q⁺ = explicit_transport(i⁺, j⁺, k⁺, grid, dir, ℓ, advection, td, u)
    q⁻ = explicit_transport(i⁻, j⁻, k⁻, grid, dir, ℓ, advection, td, u)
    cᴸ = @inbounds c[i⁻, j⁻, k⁻]
    cᴴ = @inbounds c[i, j, k]
    cᴰ = ifelse(q > 0, cᴸ, cᴴ)
    A  = q * (ψᶠ - cᴰ)

    cᴸᴸ = @inbounds c[i⁻⁻, j⁻⁻, k⁻⁻]
    cᴴᴴ = @inbounds c[i⁺, j⁺, k⁺]

    α = advection.bounds.downwind_weight
    E = @inbounds advection.bounds.extrema[1]
    m, M = E

    # beyond the extrema, a cell mean rises 1/8 of the second difference of a parabola as its extremum reaches the cell center,
    # capped by how far both cells of the face are beyond the extrema, which bounds the second difference of cells clipped to them;
    # the two second differences have opposite signs at a step, and a wall next to the two cells leaves no parabola
    flowing = !(iszero(q⁻) | iszero(q⁺))
    δ²ᴸ = cᴸᴸ + cᴴ - 2cᴸ
    δ²ᴴ = cᴸ + cᴴᴴ - 2cᴴ
    ε⁻ = ifelse(flowing, positive_part(min_fast(δ²ᴸ, min_fast(δ²ᴴ, m - max_fast(cᴸ, cᴴ)))) / 8, zero(q))
    ε⁺ = ifelse(flowing, positive_part(min_fast(-δ²ᴸ, min_fast(-δ²ᴴ, min_fast(cᴸ, cᴴ) - M))) / 8, zero(q))

    n = sum(T -> T !== Flat, topology(grid))
    τ = n * _unwrap_for_gpu(advection.bounds.Δτ)
    θᴸ = antidiffusive_fraction(cᴸ, τ * V⁻¹ᶜᶜᶜ(i⁻, j⁻, k⁻, grid), A, (-q, cᴴ), ( q⁻, cᴸᴸ), A > 0, α, E, (ε⁻, ε⁺))
    θᴴ = antidiffusive_fraction(cᴴ, τ * V⁻¹ᶜᶜᶜ(i,  j,  k,  grid), A, ( q, cᴸ), (-q⁺, cᴴᴴ), A < 0, α, E, (ε⁻, ε⁺))

    return q * (cᴰ + min_fast(θᴸ, θᴴ) * (ψᶠ - cᴰ))
end
