#####
##### Upwind-biased 3rd-order advection scheme
#####

"""
    struct UpwindBiasedFifthOrder <: AbstractUpwindBiasedAdvectionScheme{3}

Upwind-biased fifth-order advection scheme.
"""
struct UpwindBiased{N, CA, SI} <: AbstractUpwindBiasedAdvectionScheme{N} 
    "reconstruction scheme used near boundaries"
    boundary_scheme :: CA
    "reconstruction scheme used for symmetric interpolation"
    symmetric_scheme :: SI

    function UpwindBiased{N}(boundary_scheme::CA, symmetric_scheme::SI) where {N, CA, SI}
        return new{N, CA, SI}(boundary_scheme, symmetric_scheme)
    end
end

function UpwindBiased(; order = 5) 

    N  = Int((order + 1) ÷ 2)

    if N > 1
        symmetric_scheme = Centered(order = order - 1)
        boundary_scheme = UpwindBiased(order = order - 2)
    else
        symmetric_scheme = CenteredSecondOrder()
        boundary_scheme = nothing
    end

    return UpwindBiased{N}(boundary_scheme, symmetric_scheme)
end


Base.summary(a::UpwindBiased{N}) where N = string("Upwind Biased reconstruction order ", N*2-1)

Base.show(io::IO, a::UpwindBiased{N}) where {N} =
    print(io, summary(a), " \n",
              " Boundary scheme : ", "\n",
              "    └── ", summary(a.boundary_scheme) , "\n",
              " Symmetric scheme : ", "\n",
              "    └── ", summary(a.symmetric_scheme))

# Usefull aliases
UpwindBiasedFirstOrder() = UpwindBiased(order = 1)
UpwindBiasedThirdOrder() = UpwindBiased(order = 3)
UpwindBiasedFifthOrder() = UpwindBiased(order = 5)

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::AbstractUpwindBiasedAdvectionScheme, c) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.symmetric_scheme, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::AbstractUpwindBiasedAdvectionScheme, c) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.symmetric_scheme, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::AbstractUpwindBiasedAdvectionScheme, c) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.symmetric_scheme, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::AbstractUpwindBiasedAdvectionScheme, u) = symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.symmetric_scheme, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::AbstractUpwindBiasedAdvectionScheme, v) = symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.symmetric_scheme, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::AbstractUpwindBiasedAdvectionScheme, w) = symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme.symmetric_scheme, w)

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