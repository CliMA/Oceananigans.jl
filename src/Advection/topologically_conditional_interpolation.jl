#####
##### This file provides functions that conditionally-evaluate interpolation operators
##### near boundaries in bounded directions.
#####
##### For example, the function _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c) either
#####
#####     1. Always returns symmetric_interpolate_xᶠᵃᵃ if the x-direction is Periodic; or
#####
#####     2. Returns symmetric_interpolate_xᶠᵃᵃ if the x-direction is Bounded and index i is not
#####        close to the boundary, or a second-order interpolation if i is close to a boundary.
#####

using Oceananigans.Grids: AbstractUnderlyingGrid, Bounded

const AUG = AbstractUnderlyingGrid

# Left-biased buffers are smaller by one grid point on the right side; vice versa for right-biased buffers
                                                                              # outside left | outside right buffer
@inline    outside_symmetric_buffer(i, N, ::AbstractAdvectionScheme{Nᴮ}) where Nᴮ = i > Nᴮ     && i < N + 1 - Nᴮ
@inline  outside_left_biased_buffer(i, N, ::AbstractAdvectionScheme{Nᴮ}) where Nᴮ = i > Nᴮ     && i < N + 1 - (Nᴮ - 1)
@inline outside_right_biased_buffer(i, N, ::AbstractAdvectionScheme{Nᴮ}) where Nᴮ = i > Nᴮ - 1 && i < N + 1 - Nᴮ

const ADV = AbstractAdvectionScheme
const WVI = WENOVectorInvariant

for bias in (:symmetric, :left_biased, :right_biased)

    for (d, ξ) in enumerate((:x, :y, :z))

        code = [:ᵃ, :ᵃ, :ᵃ]

        for loc in (:ᶜ, :ᶠ)
            code[d] = loc
            second_order_interp = Symbol(:ℑ, ξ, code...)
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            alt_interp = Symbol(:_, interp)

            # Simple translation for Periodic directions (fallback)
            @eval $alt_interp(i, j, k, grid::AUG, scheme::ADV, args...) = $interp(i, j, k, grid, scheme, args...)

            outside_buffer = Symbol(:outside_, bias, :_buffer)

            # Conditional high-order interpolation in Bounded directions
            if ξ == :x
                @eval begin
                    @inline $alt_interp(i, j, k, grid::AUG{FT, <:Bounded}, scheme::ADV, ψ) where FT =
                        ifelse($outside_buffer(i, grid.Nx, scheme),
                               $interp(i, j, k, grid, scheme, ψ),
                               $second_order_interp(i, j, k, grid, ψ))

                    @inline $alt_interp(i, j, k, grid::AUG{FT, <:Bounded}, scheme::WVI, ζ, VI, u, v) where FT =
                        ifelse($outside_buffer(i, grid.Nx, scheme),
                            $interp(i, j, k, grid, scheme, ζ, VI, u, v),
                            $second_order_interp(i, j, k, grid, ζ, u, v))
                end
            elseif ξ == :y
                @eval begin
                    @inline $alt_interp(i, j, k, grid::AUG{FT, TX, <:Bounded}, scheme::ADV, ψ) where {FT, TX} =
                        ifelse($outside_buffer(j, grid.Ny, scheme),
                               $interp(i, j, k, grid, scheme, ψ),
                               $second_order_interp(i, j, k, grid, ψ))

                    @inline $alt_interp(i, j, k, grid::AUG{FT, TX, <:Bounded}, scheme::WVI, ζ, VI, u, v) where {FT, TX} =
                        ifelse($outside_buffer(j, grid.Ny, scheme),
                               $interp(i, j, k, grid, scheme, ζ, VI, u, v),
                               $second_order_interp(i, j, k, grid, ζ, u, v))
                end
            elseif ξ == :z
                @eval begin
                    @inline $alt_interp(i, j, k, grid::AUG{FT, TX, TY, <:Bounded}, scheme::ADV, ψ) where {FT, TX, TY} =
                        ifelse($outside_buffer(k, grid.Nz, scheme),
                               $interp(i, j, k, grid, scheme, ψ),
                               $second_order_interp(i, j, k, grid, ψ))

                    @inline $alt_interp(i, j, k, grid::AUG{FT, TX, TY, <:Bounded}, scheme::WVI, ∂z, VI, u) where {FT, TX, TY} =
                        ifelse($outside_buffer(k, grid.Nz, scheme),
                                $interp(i, j, k, grid, scheme, ∂z, VI, u),
                                $second_order_interp(i, j, k, grid, ∂z, u))
                end
            end
        end
    end
end
