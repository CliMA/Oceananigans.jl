#####
##### Note: we call this "interpolation", but these operators actually "reconstruct"
##### cell-averaged fields at _staggered_ locations.
#####

using Oceananigans.Grids: Flat
using Oceananigans.Grids: idxᴸ, idxᴿ, flip

#####
##### Base interpolation operators without grid
#####

@inline ℑxᶜᵃᵃ(i, j, k, u) = @inbounds (u[i,   j, k] + u[i+1, j, k]) / 2
@inline ℑxᶠᵃᵃ(i, j, k, c) = @inbounds (c[i-1, j, k] + c[i,   j, k]) / 2

@inline ℑyᵃᶜᵃ(i, j, k, v) = @inbounds (v[i, j,   k] + v[i,  j+1, k]) / 2
@inline ℑyᵃᶠᵃ(i, j, k, c) = @inbounds (c[i, j-1, k] + c[i,  j,   k]) / 2

@inline ℑzᵃᵃᶜ(i, j, k, w) = @inbounds (w[i, j,   k] + w[i, j, k+1]) / 2
@inline ℑzᵃᵃᶠ(i, j, k, c) = @inbounds (c[i, j, k-1] + c[i, j,   k]) / 2

#####
##### Base interpolation operators
#####

@inline ℑxᶜᵃᵃ(i, j, k, grid::AG{FT}, u) where FT = @inbounds FT(0.5) * (u[i,   j, k] + u[i+1, j, k])
@inline ℑxᶠᵃᵃ(i, j, k, grid::AG{FT}, c) where FT = @inbounds FT(0.5) * (c[i-1, j, k] + c[i,   j, k])

@inline ℑyᵃᶜᵃ(i, j, k, grid::AG{FT}, v) where FT = @inbounds FT(0.5) * (v[i, j,   k] + v[i,  j+1, k])
@inline ℑyᵃᶠᵃ(i, j, k, grid::AG{FT}, c) where FT = @inbounds FT(0.5) * (c[i, j-1, k] + c[i,  j,   k])

@inline ℑzᵃᵃᶜ(i, j, k, grid::AG{FT}, w) where FT = @inbounds FT(0.5) * (w[i, j,   k] + w[i, j, k+1])
@inline ℑzᵃᵃᶠ(i, j, k, grid::AG{FT}, c) where FT = @inbounds FT(0.5) * (c[i, j, k-1] + c[i, j,   k])

#####
##### Interpolation operators acting on functions
#####

@inline ℑxᶜᵃᵃ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i,   j, k, grid, args...) + f(i+1, j, k, grid, args...))
@inline ℑxᶠᵃᵃ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i-1, j, k, grid, args...) + f(i,   j, k, grid, args...))

@inline ℑyᵃᶜᵃ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j,   k, grid, args...) + f(i, j+1, k, grid, args...))
@inline ℑyᵃᶠᵃ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j-1, k, grid, args...) + f(i, j,   k, grid, args...))

@inline ℑzᵃᵃᶜ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j, k,   grid, args...) + f(i, j, k+1, grid, args...))
@inline ℑzᵃᵃᶠ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j, k-1, grid, args...) + f(i, j, k,   grid, args...))

#####
##### Convenience operators for "interpolating constants"
#####

@inline ℑxᶠᵃᵃ(i, j, k, grid::AG, f::Number, args...) = f
@inline ℑxᶜᵃᵃ(i, j, k, grid::AG, f::Number, args...) = f
@inline ℑyᵃᶠᵃ(i, j, k, grid::AG, f::Number, args...) = f
@inline ℑyᵃᶜᵃ(i, j, k, grid::AG, f::Number, args...) = f
@inline ℑzᵃᵃᶠ(i, j, k, grid::AG, f::Number, args...) = f
@inline ℑzᵃᵃᶜ(i, j, k, grid::AG, f::Number, args...) = f

#####
##### Leftover
#####

@inline ℑxᶠᵃᵃ(i, j, k, grid::AG{FT}, c, args...) where FT = @inbounds FT(0.5) * (c[i-1, j, k] + c[i, j, k])
@inline ℑyᵃᶠᵃ(i, j, k, grid::AG{FT}, c, args...) where FT = @inbounds FT(0.5) * (c[i, j-1, k] + c[i, j, k])
@inline ℑzᵃᵃᶠ(i, j, k, grid::AG{FT}, c, args...) where FT = @inbounds FT(0.5) * (c[i, j, k-1] + c[i, j, k])

#####
##### Location'd interpolation
#####

const c = Center()
const f = Face()

@inline onehalf(grid::AG{FT}) where FT = FT(0.5)

# `getξᴰ` retrieves staggered values along the ξ axis, in the ᴰ direction,
# where ξ is (x, y, z) and ᴰ is ᴿ for "right" or ᴸ for "left".
#
# If the location being retrieved is _inactive_, then the value on the
# _opposite_ side of the inactive is returned. Otherwise, the value
# (which is known to be active) is returned.
@inline getxᴿ(i, j, k, grid, ℓx, ℓy, ℓz, q) = @inbounds ifelse(inactive_node(idxᴿ(i, ℓx), j, k, flip(ℓx), ℓy, ℓz), q[idxᴸ(i, ℓx), j, k], q[idxᴿ(i, ℓx), j, k])
@inline getxᴸ(i, j, k, grid, ℓx, ℓy, ℓz, q) = @inbounds ifelse(inactive_node(idxᴸ(i, ℓx), j, k, flip(ℓx), ℓy, ℓz), q[idxᴿ(i, ℓx), j, k], q[idxᴸ(i, ℓx), j, k])
@inline getyᴿ(i, j, k, grid, ℓx, ℓy, ℓz, q) = @inbounds ifelse(inactive_node(i, idxᴿ(j, ℓy), k, ℓx, flip(ℓy), ℓz), q[i, idxᴸ(j, ℓy), k], q[i, idxᴿ(j, ℓy), k])
@inline getyᴸ(i, j, k, grid, ℓx, ℓy, ℓz, q) = @inbounds ifelse(inactive_node(i, idxᴸ(j, ℓy), k, ℓx, flip(ℓy), ℓz), q[i, idxᴿ(j, ℓy), k], q[i, idxᴸ(j, ℓy), k])
@inline getzᴿ(i, j, k, grid, ℓx, ℓy, ℓz, q) = @inbounds ifelse(inactive_node(i, j, idxᴿ(i, ℓx), ℓx, ℓy, flip(ℓz)), q[i, j, idxᴸ(k, ℓz)], q[i, j, idxᴿ(k, ℓz)])
@inline getzᴸ(i, j, k, grid, ℓx, ℓy, ℓz, q) = @inbounds ifelse(inactive_node(i, j, idxᴸ(i, ℓx), ℓx, ℓy, flip(ℓz)), q[i, j, idxᴿ(k, ℓz)], q[i, j, idxᴸ(k, ℓz)])

@inline getxᴿ(i, j, k, g, ℓx, ℓy, ℓz, f::F, a...) where F<:Function = ifelse(inactive_node(idxᴿ(i, ℓx), j, k, flip(ℓx), ℓy, ℓz), f(idxᴸ(i, ℓx), j, k, g, a...), f(idxᴿ(i, ℓx), j, k, g, a...))
@inline getxᴸ(i, j, k, g, ℓx, ℓy, ℓz, f::F, a...) where F<:Function = ifelse(inactive_node(idxᴸ(i, ℓx), j, k, flip(ℓx), ℓy, ℓz), f(idxᴿ(i, ℓx), j, k, g, a...), f(idxᴸ(i, ℓx), j, k, g, a...))
@inline getyᴿ(i, j, k, g, ℓx, ℓy, ℓz, f::F, a...) where F<:Function = ifelse(inactive_node(i, idxᴿ(j, ℓy), k, ℓx, flip(ℓy), ℓz), f(i, idxᴸ(j, ℓy), k, g, a...), f(i, idxᴿ(j, ℓy), k, g, a...))
@inline getyᴸ(i, j, k, g, ℓx, ℓy, ℓz, f::F, a...) where F<:Function = ifelse(inactive_node(i, idxᴸ(j, ℓy), k, ℓx, flip(ℓy), ℓz), f(i, idxᴿ(j, ℓy), k, g, a...), f(i, idxᴸ(j, ℓy), k, g, a...))
@inline getzᴿ(i, j, k, g, ℓx, ℓy, ℓz, f::F, a...) where F<:Function = ifelse(inactive_node(i, j, idxᴿ(i, ℓx), ℓx, ℓy, flip(ℓz)), f(i, j, idxᴸ(k, ℓz), g, a...), f(i, j, idxᴿ(k, ℓz), g, a...))
@inline getzᴸ(i, j, k, g, ℓx, ℓy, ℓz, f::F, a...) where F<:Function = ifelse(inactive_node(i, j, idxᴸ(i, ℓx), ℓx, ℓy, flip(ℓz)), f(i, j, idxᴿ(k, ℓz), g, a...), f(i, j, idxᴸ(k, ℓz), g, a...))

syms = (:ᶜ, :ᶠ)
locs = (:c, :f)

symflip(s) = s === :ᶜ ? :ᶠ : :ᶜ

for (sx, ℓx) in zip(syms, locs)
    for (sy, ℓy) in zip(syms, locs)
        for (sz, ℓz) in zip(syms, locs)

            for ξ in (:x, :y, :z)
                boundary = Symbol(ξ, :_boundary)
                ℑ = Symbol(:ℑ, ξ, sx, sy, sz)

                getᴸ = Symbol(:get, ξ, :ᴸ)
                getᴿ = Symbol(:get, ξ, :ᴿ)

                @eval begin
                    @inline $ℑ(i, j, k, grid, q) = onehalf(grid) * ($getᴸ(i, j, k, grid, $ℓx, $ℓy, $ℓz, q) + $getᴿ(i, j, k, grid, $ℓx, $ℓy, $ℓz, q))

                    @inline $ℑ(i, j, k, grid, q::F, args...) where F<:Function = onehalf(grid) * ($getᴸ(i, j, k, grid, $ℓx, $ℓy, $ℓz, q, args...) +
                                                                                                  $getᴿ(i, j, k, grid, $ℓx, $ℓy, $ℓz, q, args...))
                end
            end

            ℑx = Symbol(:ℑx, sx, sy, sz)
            ℑy = Symbol(:ℑy, sx, sy, sz)
            ℑz = Symbol(:ℑz, sx, sy, sz)

            @eval begin
                @inline $ℑx(i, j, k, grid::XFlatGrid, q) = @inbounds q[i, j, k]
                @inline $ℑx(i, j, k, grid::XFlatGrid, f::F, args...) where F<:Function = f(i, j, k, grid, args...)

                @inline $ℑy(i, j, k, grid::YFlatGrid, q) = @inbounds q[i, j, k]
                @inline $ℑy(i, j, k, grid::YFlatGrid, f::F, args...) where F<:Function = f(i, j, k, grid, args...)

                @inline $ℑz(i, j, k, grid::ZFlatGrid, q) = @inbounds q[i, j, k]
                @inline $ℑz(i, j, k, grid::ZFlatGrid, f::F, args...) where F<:Function = f(i, j, k, grid, args...)
            end

            # Double interpolation
            ℑ²xy = Symbol(:ℑxy, sx, sy, sz)
            ℑy   = Symbol(:ℑy,  sx, sy, sz)
            ℑx   = Symbol(:ℑx,  sx, symflip(sy), sz)
            @eval @inline $ℑ²xy(i, j, k, grid, f, args...) = $ℑy(i, j, k, grid, $ℑx, f, args...)

            ℑ²xz = Symbol(:ℑxz, sx, sy, sz)
            ℑz   = Symbol(:ℑz,  sx, sy, sz)
            ℑx   = Symbol(:ℑx,  sx, sy, symflip(sz))
            @eval @inline $ℑ²xz(i, j, k, grid, f, args...) = $ℑz(i, j, k, grid, $ℑx, f, args...)

            ℑ²yz = Symbol(:ℑyz, sx, sy, sz)
            ℑz   = Symbol(:ℑz, sx, sy, sz)
            ℑy   = Symbol(:ℑy, sx, sy, symflip(sz))
            @eval @inline $ℑ²yz(i, j, k, grid, f, args...) = $ℑz(i, j, k, grid, $ℑy, f, args...)

            # Triple interpolation
            ℑ³xyz = Symbol(:ℑxyz, sx, sy, sz)
            ℑz    = Symbol(:ℑz, sx, sy, sz)
            ℑy    = Symbol(:ℑy, sx, sy, symflip(sz))
            ℑx    = Symbol(:ℑx, sx, symflip(sy), symflip(sz))
            @eval @inline $ℑ³xyz(i, j, k, grid, f, args...) = $ℑz(i, j, k, grid, $ℑy, $ℑx, f, args...)
        end
    end
end

