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

using Oceananigans.Grids: AbstractGrid, Bounded

@inline outside_buffer(i, N, scheme) = i > boundary_buffer(scheme) && i < N + 1 - boundary_buffer(scheme)

for bias in (:symmetric, :left_biased, :right_biased)
    for (d, ξ) in enumerate((:x, :y, :z))

        code = [:ᵃ, :ᵃ, :ᵃ]

        for loc in (:ᶜ, :ᶠ)
            code[d] = loc
            second_order_interp = Symbol(:ℑ, ξ, code...)
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            alt_interp = Symbol(:_, interp)

            # Simple translation for Periodic directions (fallback)
            @eval $alt_interp(i, j, k, grid, scheme, ψ) = $interp(i, j, k, grid, scheme, ψ)

            # Conditional high-order interpolation in Bounded directions
            if ξ == :x
                @eval begin
                    @inline $alt_interp(i, j, k, grid::AbstractGrid{FT, <:Bounded}, scheme, ψ) where FT =
                        outside_buffer(i, grid.Nx, scheme) ? $interp(i, j, k, grid, scheme, ψ) : $second_order_interp(i, j, k, grid, ψ)
                end
            elseif ξ == :y
                @eval begin
                    @inline $alt_interp(i, j, k, grid::AbstractGrid{FT, TX, <:Bounded}, scheme, ψ) where {FT, TX} =
                        outside_buffer(j, grid.Ny, scheme) ? $interp(i, j, k, grid, scheme, ψ) : $second_order_interp(i, j, k, grid, ψ)
                end
            elseif ξ == :z
                @eval begin
                    @inline $alt_interp(i, j, k, grid::AbstractGrid{FT, TX, TY, <:Bounded}, scheme, ψ) where {FT, TX, TY} =
                        outside_buffer(k, grid.Nz, scheme) ? $interp(i, j, k, grid, scheme, ψ) : $second_order_interp(i, j, k, grid, ψ)
                end
            end
        end
    end
end
