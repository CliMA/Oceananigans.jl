# Translation for periodic biasections, logic for bounded biasections

@inline outside_buffer(i, N, scheme) = i > halo_buffer(scheme) && i < N + 1 - halo_buffer(scheme)

for bias in (:symmetric, :left_biased, :right_biased)

    altbias = Symbol(:_, bias)

    @eval begin
        @inline $(altbias)_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u) = $(bias)_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u)
        @inline $(altbias)_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c) = $(bias)_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)

        @inline $(altbias)_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v) = $(bias)_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v)
        @inline $(altbias)_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c) = $(bias)_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c)

        @inline $(altbias)_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w) = $(bias)_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w)
        @inline $(altbias)_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c) = $(bias)_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c)

        @inline $(altbias)_interpolate_xᶜᵃᵃ(i, j, k, grid::AbstractGrid{FT, <:Bounded},         scheme, u) where FT           = ifelse(outside_buffer(i, grid.Nx, scheme), $(bias)_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u), ℑxᶜᵃᵃ(i, j, k, grid, u)) 
        @inline $(altbias)_interpolate_xᶠᵃᵃ(i, j, k, grid::AbstractGrid{FT, <:Bounded},         scheme, c) where FT           = ifelse(outside_buffer(i, grid.Nx, scheme), $(bias)_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c), ℑxᶠᵃᵃ(i, j, k, grid, c))

        @inline $(altbias)_interpolate_yᵃᶜᵃ(i, j, k, grid::AbstractGrid{FT, TX, <:Bounded},     scheme, v) where {FT, TX}     = ifelse(outside_buffer(j, grid.Ny, scheme), $(bias)_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v), ℑyᵃᶜᵃ(i, j, k, grid, v))
        @inline $(altbias)_interpolate_yᵃᶠᵃ(i, j, k, grid::AbstractGrid{FT, TX, <:Bounded},     scheme, c) where {FT, TX}     = ifelse(outside_buffer(j, grid.Ny, scheme), $(bias)_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c), ℑyᵃᶠᵃ(i, j, k, grid, c))

        @inline $(altbias)_interpolate_zᵃᵃᶜ(i, j, k, grid::AbstractGrid{FT, TX, TY, <:Bounded}, scheme, w) where {FT, TX, TY} = ifelse(outside_buffer(k, grid.Nz, scheme), $(bias)_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w), ℑzᵃᵃᶜ(i, j, k, grid, w))
        @inline $(altbias)_interpolate_zᵃᵃᶠ(i, j, k, grid::AbstractGrid{FT, TX, TY, <:Bounded}, scheme, c) where {FT, TX, TY} = ifelse(outside_buffer(k, grid.Nz, scheme), $(bias)_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c), ℑzᵃᵃᶠ(i, j, k, grid, c))
    end
end
