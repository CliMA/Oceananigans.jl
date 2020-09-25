# Translation for periodic biasections, logic for bounded biasections

using Oceananigans.Grids: AbstractGrid, Bounded

@inline outside_buffer(i, N, scheme) = i > halo_buffer(scheme) && i < N + 1 - halo_buffer(scheme)

for bias in (:symmetric, :left_biased, :right_biased)
    for (d, ξ) in enumerate((:x, :y, :z))

        code = [:ᵃ, :ᵃ, :ᵃ]

        for loc in (:ᶜ, :ᶠ)
            code[d] = loc
            second_order_interp = Symbol(:ℑ, ξ, code...)
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            alt_interp = Symbol(:_, interp)

            @eval $alt_interp(i, j, k, grid, scheme, ψ) = $interp(i, j, k, grid, scheme, ψ)

            if ξ == :x
                @eval begin
                    $alt_interp(i, j, k, grid::AbstractGrid{FT, <:Bounded}, scheme, ψ) where FT =
                        ifelse(outside_buffer(i, grid.Nx, scheme), $interp(i, j, k, grid, scheme, ψ), $second_order_interp(i, j, k, grid, ψ))
                end
            elseif ξ == :y
                @eval begin
                    $alt_interp(i, j, k, grid::AbstractGrid{FT, TX, <:Bounded}, scheme, ψ) where {FT, TX} =
                        ifelse(outside_buffer(j, grid.Ny, scheme), $interp(i, j, k, grid, scheme, ψ), $second_order_interp(i, j, k, grid, ψ))
                end
            elseif ξ == :z
                @eval begin
                    $alt_interp(i, j, k, grid::AbstractGrid{FT, TX, TY, <:Bounded}, scheme, ψ) where {FT, TX, TY} =
                        ifelse(outside_buffer(k, grid.Nz, scheme), $interp(i, j, k, grid, scheme, ψ), $second_order_interp(i, j, k, grid, ψ))
                end
            end
        end
    end
end
