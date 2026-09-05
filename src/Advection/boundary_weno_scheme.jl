using Oceananigans.ImmersedBoundaries: inactive_node
using Oceananigans.Operators: Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ

"""
    CWENOZ([FT=Oceananigans.defaults.FloatType;] reference_gradient = 0)

Third order central-WENO reconstruction on a stencil that extends only into the interior, for use as the
`boundary_scheme` of a `DecreasingOrderAdvectionScheme`. An inward parabola, a linear polynomial and a constant
are blended with Z-weights [SempliceTravagliaPuppo22](@cite).

- `reference_gradient`: gradient of the reconstructed field, in units of `ψ` per unit length, setting the scale
  `ϵ = (reference_gradient * Δ)²` at which the constant takes over. Raising it activates the constant less
  often, which lowers the spurious mixing against the boundary and raises the overshoot; lowering it does the
  reverse. Zero estimates it from the stencil.
"""
struct CWENOZ{FT, C}
    reference_gradient :: FT
    symmetric_scheme :: C
end

CWENOZ(FT::DataType = Oceananigans.defaults.FloatType; reference_gradient = 0) = CWENOZ(convert(FT, reference_gradient), Centered(FT; order=2))

Base.eltype(::CWENOZ{FT}) where FT = FT
Base.summary(scheme::CWENOZ{FT}) where FT = string("CWENOZ{$FT}(reference_gradient=", scheme.reference_gradient, ")")
Base.show(io::IO, scheme::CWENOZ) = print(io, summary(scheme))

@inline  inward_parabola_oscillation(u₁, u₂, u₃) = @muladd 13//12 * (u₁ - 2u₂ + u₃)^2 + 1//4 * (3u₁ - 4u₂ + u₃)^2
@inline centred_parabola_oscillation(u₁, u₂, u₃) = @muladd 13//12 * (u₁ - 2u₂ + u₃)^2 + 1//4 * (u₁ - u₃)^2
@inline linear_oscillation(uₐ, u_b) = (u_b - uₐ)^2

@inline function boundary_oscillation_floor(u₁, u₂, u₃, Δ, ∇ref)
    estimated = min(linear_oscillation(u₁, u₂), linear_oscillation(u₂, u₃))
    ϵ = ifelse(∇ref > zero(∇ref), (∇ref * Δ)^2, estimated)
    return max(ϵ, convert(typeof(u₁), 1e-14) * max(abs(u₁), abs(u₂), abs(u₃))^2)
end

@inline smoothness_ratio(τ, β, ϵ) = ifelse(β + ϵ > zero(τ), τ / (β + ϵ), zero(τ))

"""
    cwenoz_reconstruction(scheme, u₁, u₂, u₃, wet₂, wet₃, Δ)

Point value at the inward face of the boundary cell, from the cell averages `u₁, u₂, u₃` running inwards from
the boundary. `wet₂` and `wet₃` report whether the second and third cells exist; a run too short for a candidate
drops to the one that fits.
"""
@inline function cwenoz_reconstruction(scheme::CWENOZ{FT}, u₁, u₂, u₃, wet₂, wet₃, Δ) where FT
    d⁰, d¹ = convert(FT, 0.01), convert(FT, 0.25)
    dᵒ = 1 - d¹ - d⁰

    u₂ = ifelse(wet₂, u₂, u₁)
    u₃ = ifelse(wet₃, u₃, u₂)

    ϵ  = boundary_oscillation_floor(u₁, u₂, u₃, Δ, scheme.reference_gradient)
    I² = inward_parabola_oscillation(u₁, u₂, u₃)
    I¹ = linear_oscillation(u₁, u₂)

    # τ is the indicator of the first interior cell, whose stencil is these same three averages
    τ = abs(2 * centred_parabola_oscillation(u₁, u₂, u₃) - linear_oscillation(u₁, u₂) - linear_oscillation(u₂, u₃))

    αᵒ = dᵒ * (1 + smoothness_ratio(τ, I², ϵ))
    α¹ = d¹ * (1 + smoothness_ratio(τ, I¹, ϵ))
    α⁰ = d⁰ * (1 + smoothness_ratio(τ, zero(FT), ϵ))
    Σα = αᵒ + α¹ + α⁰

    P⁰ = u₁
    P¹ = (u₁ + u₂) / 2
    P² = (u₁ + 5u₂ / 2 - u₃ / 2) / 3

    Pᵒ = @muladd (P² - d¹ * P¹ - d⁰ * P⁰) * (1 / dᵒ)
    blended = @muladd (αᵒ * Pᵒ + α¹ * P¹ + α⁰ * P⁰) / Σα

    return ifelse(wet₃, blended, ifelse(wet₂, P¹, u₁))
end

for (d, ξ) in enumerate((:x, :y, :z))
    code = [:ᵃ, :ᵃ, :ᵃ]

    for loc in (:ᶜ, :ᶠ)
        code[d] = loc
        interp = Symbol(:biased_interpolate_, ξ, code...)

        # a `ᶜ` reconstruction is the `ᶠ` operator one index along, reading a Face-located field
        halfshift = loc == :ᶠ ? 0 : 1

        # Four constant offsets spanning both biases, so the loads have static addresses: `LeftBias`
        # reads (J₋₁, J₀, J₊₁) inwards and `RightBias` reads (J₀, J₋₁, J₋₂).
        at(o) = ξ == :x ? (:($(:i) + $o), :j, :k) :
                ξ == :y ? (:i, :($(:j) + $o), :k) : (:i, :j, :($(:k) + $o))
        J₋₂, J₋₁, J₀, J₊₁ = at(-2 + halfshift), at(-1 + halfshift), at(halfshift), at(1 + halfshift)

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

            # only a wet run shorter than three cells needs these
            wet₂ = ifelse(inward, !inactive_node($(J₀...),  grid, $ℓx, $ℓy, $ℓz),
                                  !inactive_node($(J₋₁...), grid, $ℓx, $ℓy, $ℓz))
            wet₃ = ifelse(inward, !inactive_node($(J₊₁...), grid, $ℓx, $ℓy, $ℓz),
                                  !inactive_node($(J₋₂...), grid, $ℓx, $ℓy, $ℓz))

            return cwenoz_reconstruction(scheme, u₁, u₂, u₃, wet₂, wet₃, $(Symbol(:Δ, ξ, :ᶜᶜᶜ))($(J₀...), grid))
        end
    end
end
