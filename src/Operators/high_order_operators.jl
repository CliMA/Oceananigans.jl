using Base: Callable
using MuladdMacro: @muladd
using Oceananigans.Grids: Flat, Open, Periodic, RegularVerticalCoordinate, TwoSided, inactive_node

#####
##### Interpolations and differences of order `O`, with coefficients from finite-volume reconstruction.
##### Operators without an order argument are second order.
#####

const NonFlat = Union{Periodic, TwoSided}

for (ξ, RegularGrid) in ((:x, :(RG{<:Any, <:NonFlat, <:Any, <:Any, <:Any, <:Number})),
                         (:y, :(RG{<:Any, <:Any, <:NonFlat, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number})),
                         (:z, :(RG{<:Any, <:Any, <:Any, <:NonFlat, <:RegularVerticalCoordinate})))

    ℑ = Symbol(:ℑ, ξ, insert_symbol(ξ, :ᶠ, :ᵃ, :ᵃ)...)

    @eval @generated function $ℑ(i, j, k, grid::AG{FT}, ::Val{O}, ψ, args...) where {FT, O}
        stencil = calc_reconstruction_stencil(FT, O ÷ 2, :symmetric, $(QuoteNode(ξ)), ψ <: Callable)
        return :($(Expr(:meta, :inline)); @inbounds @muladd $stencil)
    end

    for location in (:ᶠ, :ᶜ)
        δ = Symbol(:δ, ξ, insert_symbol(ξ, location, :ᵃ, :ᵃ)...)

        @eval begin
            @inline $δ(i, j, k, grid, ::Val, ψ) = $δ(i, j, k, grid, ψ)

            @generated function $δ(i, j, k, grid::$RegularGrid, ::Val{O}, ψ) where O
                stencil = calc_reconstruction_stencil(eltype(grid), O ÷ 2, Val(:first_derivative), $(QuoteNode(ξ)), $(QuoteNode(location)))
                return :($(Expr(:meta, :inline)); @inbounds @muladd $stencil)
            end
        end
    end
end

#####
##### Located differences and derivatives: the order decreases near boundaries
#####

"""
    near_boundary(i, j, k, grid, ::Val{O}, ::Val{d}, ℓx, ℓy, ℓz)

Return `true` if the `O`-point stencil in direction `d` around `i, j, k`, reading a field at `(ℓx, ℓy, ℓz)`,
includes an `inactive_node` or a node beyond an `Open` side.
"""
@generated function near_boundary(i, j, k, grid::AG{<:Any, TX, TY, TZ}, ::Val{O}, ::Val{d}, ℓx, ℓy, ℓz) where {TX, TY, TZ, O, d}
    topology = (TX, TY, TZ)[d]
    topology <: Flat && return :(false)

    location = (ℓx, ℓy, ℓz)[d]
    first_node, last_node = location <: Center ? (-(O ÷ 2), O ÷ 2 - 1) : (1 - O ÷ 2, O ÷ 2)
    index = (:i, :j, :k)[d]
    conditions = Any[]

    for s in first_node:last_node
        node = Any[:i, :j, :k]
        node[d] = :($index + $s)
        push!(conditions, :(inactive_node($(node...), grid, ℓx, ℓy, ℓz)))
    end

    N = (:Nx, :Ny, :Nz)[d]
    topology <: TwoSided{Open} && push!(conditions, :($index + $first_node < 1))
    topology <: TwoSided{<:Any, Open} && push!(conditions, :($index + $last_node > grid.$N + $(location <: Face)))

    return :($(Expr(:meta, :inline)); (|)($(conditions...)))
end

for (d, ξ) in enumerate((:x, :y, :z)), ℓ1 in (:ᶜ, :ᶠ), ℓ2 in (:ᶜ, :ᶠ), ℓ3 in (:ᶜ, :ᶠ)
    location = insert_symbol(ξ, ℓ1, ℓ2, ℓ3)
    ℓx, ℓy, ℓz = map(ℓ -> ℓ == :ᶜ ? :c : :f, insert_symbol(ξ, ℓ1 == :ᶜ ? :ᶠ : :ᶜ, ℓ2, ℓ3))

    δ   = Symbol(:δ, ξ, location...)
    δᵃ  = Symbol(:δ, ξ, insert_symbol(ξ, ℓ1, :ᵃ, :ᵃ)...)
    ∂   = d == 3 ? Symbol(:∂z, location...) : Symbol(:∂, ξ, :ᵣ, location...)
    Δ⁻¹ = Symbol(:Δ, ξ, :⁻¹, location...)

    @eval begin
        @inline $δ(i, j, k, grid, ::Val{2}, φ) = $δ(i, j, k, grid, φ)
        @inline $δ(i, j, k, grid, ::Val{O}, φ) where O = ifelse(near_boundary(i, j, k, grid, Val(O), Val($d), $ℓx, $ℓy, $ℓz), $δ(i, j, k, grid, Val(O - 2), φ), $δᵃ(i, j, k, grid, Val(O), φ))
        @inline $∂(i, j, k, grid, order::Val, φ) = $δ(i, j, k, grid, order, φ) * $Δ⁻¹(i, j, k, grid)
    end
end
