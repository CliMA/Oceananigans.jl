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

using Oceananigans.Grids: AbstractPrimaryGrid, Bounded

const APG = AbstractPrimaryGrid

# Left-biased buffers are smaller by one grid point on the right side; vice versa for right-biased buffers
                                                    # outside left buffer          outside right buffer
@inline outside_symmetric_buffer(i, N, scheme)    = i > boundary_buffer(scheme)     && i < N + 1 - boundary_buffer(scheme)
@inline outside_left_biased_buffer(i, N, scheme)  = i > boundary_buffer(scheme)     && i < N + 1 - (boundary_buffer(scheme) - 1)
@inline outside_right_biased_buffer(i, N, scheme) = i > boundary_buffer(scheme) - 1 && i < N + 1 - boundary_buffer(scheme)

for bias in (:symmetric, :left_biased, :right_biased)

    for (d, ξ) in enumerate((:x, :y, :z))

        code = [:ᵃ, :ᵃ, :ᵃ]

        for loc in (:ᶜ, :ᶠ)
            code[d] = loc
            second_order_interp = Symbol(:ℑ, ξ, code...)
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            alt_interp = Symbol(:_, interp)

            # Simple translation for Periodic directions (fallback)
            @eval $alt_interp(i, j, k, grid::APG, scheme, ψ) = $interp(i, j, k, grid, scheme, ψ)

            outside_buffer = Symbol(:outside, bias, :_buffer)

            # Conditional high-order interpolation in Bounded directions
            if ξ == :x
                @eval begin
                    @inline $alt_interp(i, j, k, grid::APG{FT, <:Bounded}, scheme, ψ) where FT =
                        ifelse($outside_buffer(i, grid.Nx, scheme), $interp(i, j, k, grid, scheme, ψ), $second_order_interp(i, j, k, grid, ψ))
                end
            elseif ξ == :y
                @eval begin
                    @inline $alt_interp(i, j, k, grid::APG{FT, TX, <:Bounded}, scheme, ψ) where {FT, TX} =
                        ifelse($outside_buffer(j, grid.Ny, scheme), $interp(i, j, k, grid, scheme, ψ), $second_order_interp(i, j, k, grid, ψ))
                end
            elseif ξ == :z
                @eval begin
                    @inline $alt_interp(i, j, k, grid::APG{FT, TX, TY, <:Bounded}, scheme, ψ) where {FT, TX, TY} =
                        ifelse($outside_buffer(k, grid.Nz, scheme), $interp(i, j, k, grid, scheme, ψ), $second_order_interp(i, j, k, grid, ψ))
                end
            end
        end
    end
end
