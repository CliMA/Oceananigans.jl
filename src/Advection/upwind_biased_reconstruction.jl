#####
##### Upwind-biased 3rd-order advection scheme
#####

"""
    struct UpwindBiasedFifthOrder <: AbstractUpwindBiasedAdvectionScheme{3}

Upwind-biased fifth-order advection scheme.
"""
struct UpwindBiased{N, CA, SI} <: AbstractUpwindBiasedAdvectionScheme{N} 
    "advection scheme used near boundaries"
    boundary_scheme :: CA
    symmetric_scheme :: SI
end

function UpwindBiased(; order = 5) 

    N  = Int((order + 1) ÷ 2)

    if N >= 2
        symmetric_scheme = CenteredFourthOrder()
        boundary_scheme = UpwindBiased(order = order - 2)
    elseif N == 1
        symmetric_scheme = CenteredSecondOrder()
        boundary_scheme = nothing
    end

    return UpwindBiased{N}(boundary_scheme, symmetric_scheme)
end

# Usefull aliases
UpwindBiasedFirstOrder() = UpwindBiased(order = 1)
UpwindBiasedThirdOrder() = UpwindBiased(order = 3)
UpwindBiasedFifthOrder() = UpwindBiased(order = 5)

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::UpwindBiased, c) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.symmetric_scheme, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::UpwindBiased, c) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.symmetric_scheme, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::UpwindBiased, c) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.symmetric_scheme, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::UpwindBiased, u) = symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.symmetric_scheme, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::UpwindBiased, v) = symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.symmetric_scheme, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::UpwindBiased, w) = symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme.symmetric_scheme, w)

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::UpwindBiased, u, args...) = left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u, args...)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::UpwindBiased, v, args...) = left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v, args...)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::UpwindBiased, w, args...) = left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w, args...)

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::UpwindBiased, u, args...) = right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u, args...)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::UpwindBiased, v, args...) = right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v, args...)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::UpwindBiased, w, args...) = right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w, args...)

for side in (:left, :right), (dir, ξ) in zip((:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ), (:x, :y, :z))
    stencil = Symbol(side, :_biased_interpolate_, dir)

    for buffer in [1, 2, 3, 4, 5]
        @eval begin
            @inline $stencil(i, j, k, grid, scheme::UpwindBiased{$buffer}, ψ, args...)           = @inbounds $(calc_advection_stencil(buffer, side, ξ, false))
            @inline $stencil(i, j, k, grid, scheme::UpwindBiased{$buffer}, ψ::Function, args...) = @inbounds $(calc_advection_stencil(buffer, side, ξ,  true))
        end
    end
end