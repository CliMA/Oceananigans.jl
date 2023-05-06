@inline _multi_dimensional_reconstruction_x(args...) = multi_dimensional_reconstruction_x(args...)
@inline _multi_dimensional_reconstruction_y(args...) = multi_dimensional_reconstruction_y(args...)

const two_32 = Int32(2)

## Figure them out!
const γ₀¹  = (1008 + 71 * sqrt(15)) / 5240
const γ₁¹  =  408 / 655
const γ₂¹  = (1008 - 71 * sqrt(15)) / 5240

const σ⁺ = 214/80
const σ⁻ =  67/40

const γ₀²⁺ =   9.0 / 80 / σ⁺
const γ₁²⁺ =  49.0 / 20 / σ⁺
const γ₂²⁺ =   9.0 / 80 / σ⁺

const γ₀²⁻ =   9.0 / 40 / σ⁻
const γ₁²⁻ =  49.0 / 40 / σ⁻
const γ₂²⁻ =   9.0 / 40 / σ⁻

const γ₀³  = (1008 - 71 * sqrt(15)) / 5240
const γ₁³  =  408 / 655
const γ₂³  = (1008 + 71 * sqrt(15)) / 5240

## Figure them out!
const a₀¹ = ( 2 - 3*sqrt(15), -4 + 12*sqrt(15), 62 - 9 * sqrt(15)) ./ 60
const a₁¹ = ( 2 + 3*sqrt(15),               56,  2 - 3 * sqrt(15)) ./ 60
const a₂¹ = (62 + 9*sqrt(15), -4 - 12*sqrt(15),  2 + 3 * sqrt(15)) ./ 60

const a₀² = (-1, 2,  23) ./ 24
const a₁² = (-1, 26, -1) ./ 24
const a₂² = (23, 2,  -1) ./ 24

const a₀³ = ( 2 + 3*sqrt(15), -4 - 12*sqrt(15), 62 + 9 * sqrt(15)) ./ 60
const a₁³ = ( 2 - 3*sqrt(15),               56,  2 + 3 * sqrt(15)) ./ 60
const a₂³ = (62 - 9*sqrt(15), -4 + 12*sqrt(15),  2 - 3 * sqrt(15)) ./ 60

@inline   left_biased_β_constant(FT, ψ) = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * ( ψ[1] - 4ψ[2] + 3ψ[3])^two_32
@inline center_biased_β_constant(FT, ψ) = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * ( ψ[1]         -  ψ[3])^two_32
@inline  right_biased_β_constant(FT, ψ) = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * (3ψ[1] - 4ψ[2] +  ψ[3])^two_32

@inline function centered_reconstruction_weights(FT, β₀, β₁, β₂, γ₀, γ₁, γ₂)

    α₀ = FT(γ₀) / (β₀ + FT(ε))^two_32
    α₁ = FT(γ₁) / (β₁ + FT(ε))^two_32
    α₂ = FT(γ₂) / (β₂ + FT(ε))^two_32

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα
    
    return w₀, w₁, w₂
end

function fifth_order_weno_reconstruction(FT, S₀, S₁, S₂)

    q̂₀¹ = FT(a₀¹[1]) * S₀[1] + FT(a₀¹[2]) * S₀[2] + FT(a₀¹[3]) * S₀[3]
    q̂₁¹ = FT(a₁¹[1]) * S₁[1] + FT(a₁¹[2]) * S₁[2] + FT(a₁¹[3]) * S₁[3]
    q̂₂¹ = FT(a₂¹[1]) * S₂[1] + FT(a₂¹[2]) * S₂[2] + FT(a₂¹[3]) * S₂[3]

    q̂₀² = FT(a₀²[1]) * S₀[1] + FT(a₀²[2]) * S₀[2] + FT(a₀²[3]) * S₀[3]
    q̂₁² = FT(a₁²[1]) * S₁[1] + FT(a₁²[2]) * S₁[2] + FT(a₁²[3]) * S₁[3]
    q̂₂² = FT(a₂²[1]) * S₂[1] + FT(a₂²[2]) * S₂[2] + FT(a₂²[3]) * S₂[3]
    
    q̂₀³ = FT(a₀³[1]) * S₀[1] + FT(a₀³[2]) * S₀[2] + FT(a₀³[3]) * S₀[3]
    q̂₁³ = FT(a₁³[1]) * S₁[1] + FT(a₁³[2]) * S₁[2] + FT(a₁³[3]) * S₁[3]
    q̂₂³ = FT(a₂³[1]) * S₂[1] + FT(a₂³[2]) * S₂[2] + FT(a₂³[3]) * S₂[3]

    β₀ =   left_biased_β_constant(FT, S₀)
    β₁ = center_biased_β_constant(FT, S₁)
    β₂ =  right_biased_β_constant(FT, S₂)

    w₀¹, w₁¹, w₂¹ = centered_reconstruction_weights(FT, β₀, β₁, β₂, γ₀¹, γ₁¹, γ₂¹)
    w₀³, w₁³, w₂³ = centered_reconstruction_weights(FT, β₀, β₁, β₂, γ₀³, γ₁³, γ₂³)

    w₀²⁺, w₁²⁺, w₂²⁺ = centered_reconstruction_weights(FT, β₀, β₁, β₂, γ₀²⁺, γ₁²⁺, γ₂²⁺)
    w₀²⁻, w₁²⁻, w₂²⁻ = centered_reconstruction_weights(FT, β₀, β₁, β₂, γ₀²⁻, γ₁²⁻, γ₂²⁻)

    q¹ = w₀¹ * q̂₀¹ + w₁¹ * q̂₁¹ + w₂¹ * q̂₂¹
    q³ = w₀³ * q̂₀³ + w₁³ * q̂₁³ + w₂³ * q̂₂³

    q²⁺ = w₀²⁺ * q̂₀² + w₁²⁺ * q̂₁² + w₂²⁺ * q̂₂²
    q²⁻ = w₀²⁻ * q̂₀² + w₁²⁻ * q̂₁² + w₂²⁻ * q̂₂²

    q² = FT(σ⁺) * q²⁺ - FT(σ⁻) * q²⁻

    return q¹ / 6 + 2 * q² / 3 + q³ / 6
end
@inline function multi_dimensional_reconstruction_x(i, j, k, grid, scheme, _interpolate_y, f::Function, VI::AbstractSmoothnessStencil, args...)

    FT = eltype(grid)

    Q₋₂ = _interpolate_y(i-2, j, k, grid, scheme, f, VI, args...)
    Q₋₁ = _interpolate_y(i-1, j, k, grid, scheme, f, VI, args...)
    Q₀  = _interpolate_y(i,   j, k, grid, scheme, f, VI, args...)
    Q₊₁ = _interpolate_y(i+1, j, k, grid, scheme, f, VI, args...)
    Q₊₂ = _interpolate_y(i+2, j, k, grid, scheme, f, VI, args...)

    S₀ = (Q₋₂, Q₋₁, Q₀)
    S₁ = (Q₋₁, Q₀ , Q₊₁)
    S₂ = (Q₀ , Q₊₁, Q₊₂)

    return fifth_order_weno_reconstruction(FT, S₀, S₁, S₂)
end

@inline function multi_dimensional_reconstruction_y(i, j, k, grid, scheme, _interpolate_x, f::Function, VI::AbstractSmoothnessStencil, args...)

    FT = eltype(grid)

    Q₋₂ = _interpolate_x(i, j-2, k, grid, scheme, f, VI, args...)
    Q₋₁ = _interpolate_x(i, j-1, k, grid, scheme, f, VI, args...)
    Q₀  = _interpolate_x(i, j,   k, grid, scheme, f, VI, args...)
    Q₊₁ = _interpolate_x(i, j+1, k, grid, scheme, f, VI, args...)
    Q₊₂ = _interpolate_x(i, j+2, k, grid, scheme, f, VI, args...)

    S₀ = (Q₋₂, Q₋₁, Q₀)
    S₁ = (Q₋₁, Q₀ , Q₊₁)
    S₂ = (Q₀ , Q₊₁, Q₊₂)

    return fifth_order_weno_reconstruction(FT, S₀, S₁, S₂)
end