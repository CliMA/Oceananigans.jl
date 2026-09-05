using Oceananigans.ImmersedBoundaries: inactive_node
using Oceananigans.Operators: Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ

"""
    CWENOZ([FT=Oceananigans.defaults.FloatType;]
           reference_gradient = 0,
           reference_length = 0,
           linear_weight = 0.25,
           maximum_constant_weight = 0.01,
           constant_weight_exponent = 1,
           smoothness_ratio_exponent = 1,
           relative_oscillation_floor = eps(FT),
           symmetric_scheme = Centered(FT; order=2))

Third order central-WENO reconstruction on a stencil that extends only into the interior, for use as the `boundary_scheme` of a
`DecreasingOrderAdvectionScheme`. An inward parabola `P²`, a linear polynomial `P¹` and a constant `P⁰` are blended with Z-weights:
the `CWZb3` reconstruction of [SempliceTravagliaPuppo22](@cite).

Explicitly, the reconstruction is `(αᵒPᵒ + α¹P¹ + α⁰P⁰) / (αᵒ + α¹ + α⁰)` with `αₖ = dₖ(1 + (τ / (Iₖ + ϵ))^p)`, where `Iₖ` is the oscillation
indicator of the k-th candidate and `τ` a global one. The linear weights `d¹` of `P¹` and `d⁰` of `P⁰` are free, in the sense that the
optimal polynomial is defined as `Pᵒ = (P² - d¹P¹ - d⁰P⁰) / dᵒ` with `dᵒ = 1 - d¹ - d⁰`, so that `P²` is recovered exactly on smooth data whatever
positive weights are chosen. However, they set how quickly each low order candidate takes over near a discontinuity.

- `reference_gradient`: gradient of the reconstructed field, in units of `ψ` per unit length, setting the scale `ϵ = (reference_gradient * Δ)²`
  at which the constant takes over. Raising it lowers the spurious mixing against the boundary and raises the overshoot. Zero estimates it from the stencil.

- `reference_length`: length scale that nondimensionalizes the grid spacing in `d⁰ = min((Δ / reference_length)^constant_weight_exponent, maximum_constant_weight)`.
  `d⁰ -> 0` under refinement is what recovers third order accuracy on smooth data next to the boundary. Zero holds `d⁰` at `maximum_constant_weight`.

- `linear_weight`: `d¹`, the linear weight of the linear polynomial.

- `maximum_constant_weight`: the cap on `d⁰`, and its value everywhere if `reference_length` is zero. This cap fixes how sharp a feature has to be
  before the constant stencil contributes: lowering it weights less the constant stencil against the high order one.

- `constant_weight_exponent`: the exponent with which `d⁰` vanishes under refinement. Third order accuracy needs it to be at least one.

- `smoothness_ratio_exponent`: the exponent `p` of the Z-weights, which sharpens the response to a discontinuity as it grows. Third order accuracy needs it to be at least one.

- `relative_oscillation_floor`: the value `ϵ` falls back to, relative to the square of the largest average in the stencil, where the estimate from
  the stencil vanishes. A flat pair of averages next to a jump is exactly such a case, and the constant only takes over there because `ϵ` stays
  positive: lowering the floor sharpens that response.

- `symmetric_scheme`: the reconstruction that the symmetric interpolations, which need no bias, defer to.

The defaults are those of [SempliceTravagliaPuppo22](@cite) and, through it, of [NaumannKolbSemplice18](@cite).
"""
struct CWENOZ{FT, M, P, C}
    reference_gradient :: FT
    reference_length :: FT
    linear_weight :: FT
    maximum_constant_weight :: FT
    relative_oscillation_floor :: FT
    symmetric_scheme :: C
end

function CWENOZ(FT::DataType = Oceananigans.defaults.FloatType;
                reference_gradient = 0,
                reference_length = 0,
                linear_weight = 0.25,
                maximum_constant_weight = 0.01,
                constant_weight_exponent = 1,
                smoothness_ratio_exponent = 1,
                relative_oscillation_floor = eps(FT),
                symmetric_scheme = Centered(FT; order=2))

    M = Int(constant_weight_exponent)
    P = Int(smoothness_ratio_exponent)

    return CWENOZ{FT, M, P, typeof(symmetric_scheme)}(convert(FT, reference_gradient),
                                                      convert(FT, reference_length),
                                                      convert(FT, linear_weight),
                                                      convert(FT, maximum_constant_weight),
                                                      convert(FT, relative_oscillation_floor),
                                                      symmetric_scheme)
end

Base.eltype(::CWENOZ{FT}) where FT = FT
Base.summary(::CWENOZ{FT}) where FT = string("CWENOZ{$FT}")

Base.show(io::IO, scheme::CWENOZ{FT, M, P}) where {FT, M, P} =
    print(io, summary(scheme), '\n',
              "├── reference_gradient: ",         scheme.reference_gradient, '\n',
              "├── reference_length: ",           scheme.reference_length, '\n',
              "├── linear_weight: ",              scheme.linear_weight, '\n',
              "├── maximum_constant_weight: ",    scheme.maximum_constant_weight, '\n',
              "├── constant_weight_exponent: ",   M, '\n',
              "├── smoothness_ratio_exponent: ",  P, '\n',
              "├── relative_oscillation_floor: ", scheme.relative_oscillation_floor, '\n',
              "└── symmetric_scheme: ",           summary(scheme.symmetric_scheme))

@inline  inward_parabola_oscillation(u₁, u₂, u₃) = @muladd 13//12 * (u₁ - 2u₂ + u₃)^2 + 1//4 * (3u₁ - 4u₂ + u₃)^2
@inline centred_parabola_oscillation(u₁, u₂, u₃) = @muladd 13//12 * (u₁ - 2u₂ + u₃)^2 + 1//4 * (u₁ - u₃)^2
@inline linear_oscillation(uₐ, u_b) = (u_b - uₐ)^2

@inline smoothness_ratio(scheme::CWENOZ{FT, M, P}, τ, β, ϵ) where {FT, M, P} = ifelse(β + ϵ > zero(τ), Base.literal_pow(^, τ / (β + ϵ), Val(P)), zero(τ))

"""
    cwenoz_reconstruction(scheme, u₁, u₂, u₃, active₂, active₃, Δ)

Point value at the inward face of the boundary cell, from the cell averages `u₁, u₂, u₃` running inwards from
the boundary. `active₂` and `active₃` report whether the second and third cells exist; a run too short for a
candidate drops to the one that fits.
"""
@inline function cwenoz_reconstruction(scheme::CWENOZ{FT, M}, u₁, u₂, u₃, active₂, active₃, Δ) where {FT, M}
    # Δ / 0 is infinite, so a vanishing reference length pins d⁰ to its cap
    d⁰ = min(Base.literal_pow(^, Δ / scheme.reference_length, Val(M)), scheme.maximum_constant_weight)
    d¹ = scheme.linear_weight
    dᵒ = one(FT) - d¹ - d⁰

    u₂ = ifelse(active₂, u₂, u₁)
    u₃ = ifelse(active₃, u₃, u₂)

    I² = inward_parabola_oscillation(u₁, u₂, u₃)
    I¹ = linear_oscillation(u₁, u₂)

    ∇ref = scheme.reference_gradient
    estimated = ifelse(∇ref > zero(FT), (∇ref * Δ)^2, min(I¹, linear_oscillation(u₂, u₃)))
    ϵ = max(estimated, scheme.relative_oscillation_floor * max(abs(u₁), abs(u₂), abs(u₃))^2)

    # τ is the indicator of the first interior cell, whose stencil is these same three averages
    τ = abs(2 * centred_parabola_oscillation(u₁, u₂, u₃) - I¹ - linear_oscillation(u₂, u₃))

    αᵒ = dᵒ * (one(FT) + smoothness_ratio(scheme, τ, I², ϵ))
    α¹ = d¹ * (one(FT) + smoothness_ratio(scheme, τ, I¹, ϵ))
    α⁰ = d⁰ * (one(FT) + smoothness_ratio(scheme, τ, zero(FT), ϵ))
    Σα = αᵒ + α¹ + α⁰

    P⁰ = u₁
    P¹ = (u₁ + u₂) / 2
    P² = (u₁ + 5u₂ / 2 - u₃ / 2) / 3

    Pᵒ = @muladd (P² - d¹ * P¹ - d⁰ * P⁰) / dᵒ
    blended = @muladd (αᵒ * Pᵒ + α¹ * P¹ + α⁰ * P⁰) / Σα

    return ifelse(active₃, blended, ifelse(active₂, P¹, u₁))
end

for (d, ξ) in enumerate((:x, :y, :z))
    code = [:ᵃ, :ᵃ, :ᵃ]

    for loc in (:ᶜ, :ᶠ)
        code[d] = loc
        interp = Symbol(:biased_interpolate_, ξ, code...)

        # a `ᶜ` reconstruction is the `ᶠ` operator one index along, reading a Face-located field
        halfshift = loc == :ᶠ ? 0 : 1

        shifted_indices(offset) = ξ == :x ? (:(i + $offset), :j, :k) : 
                                  ξ == :y ? (:i, :(j + $offset), :k) : 
                                            (:i, :j, :(k + $offset))

        # four offsets, so that both biases read from the same static addresses: inwards (J₋₁, J₀, J₊₁), outwards (J₀, J₋₁, J₋₂)
        J₋₂, J₋₁, J₀, J₊₁ = shifted_indices.(halfshift .+ (-2, -1, 0, 1))

        # Location of the reconstructed field.
        ℓx, ℓy, ℓz = if loc == :ᶠ
            (:c, :c, :c)
        elseif ξ == :x
            (:f, :c, :c)
        elseif ξ == :y
            (:c, :f, :c)
        else
            (:c, :c, :f)
        end

        symmetric = Symbol(:symmetric_interpolate_, ξ, code...)
        @eval @inline $symmetric(i, j, k, grid, scheme::CWENOZ, args...) = $symmetric(i, j, k, grid, scheme.symmetric_scheme, args...)

        @eval @inline function $interp(i, j, k, grid, scheme::CWENOZ, bias, ψ)
            inward = bias == LeftBias

            ψ₋₂ = @inbounds ψ[$(J₋₂...)]
            ψ₋₁ = @inbounds ψ[$(J₋₁...)]
            ψ₀  = @inbounds ψ[$(J₀...)]
            ψ₊₁ = @inbounds ψ[$(J₊₁...)]

            u₁ = ifelse(inward, ψ₋₁, ψ₀)
            u₂ = ifelse(inward, ψ₀,  ψ₋₁)
            u₃ = ifelse(inward, ψ₊₁, ψ₋₂)

            # only an active run shorter than three cells needs these
            active₂ = ifelse(inward, !inactive_node($(J₀...),  grid, $ℓx, $ℓy, $ℓz),
                                     !inactive_node($(J₋₁...), grid, $ℓx, $ℓy, $ℓz))
            active₃ = ifelse(inward, !inactive_node($(J₊₁...), grid, $ℓx, $ℓy, $ℓz),
                                     !inactive_node($(J₋₂...), grid, $ℓx, $ℓy, $ℓz))

            return cwenoz_reconstruction(scheme, u₁, u₂, u₃, active₂, active₃, $(Symbol(:Δ, ξ, :ᶜᶜᶜ))($(J₀...), grid))
        end
    end
end
