using Oceananigans.Grids: Center, Face, Flat, x_boundary, y_boundary, z_boundary

#####
##### Base difference operators
#####

@inline δxᶜᵃᵃ(i, j, k, grid, u) = @inbounds u[i+1, j, k] - u[i,   j, k]
@inline δxᶠᵃᵃ(i, j, k, grid, c) = @inbounds c[i,   j, k] - c[i-1, j, k]

@inline δyᵃᶜᵃ(i, j, k, grid, v) = @inbounds v[i, j+1, k] - v[i, j,   k]
@inline δyᵃᶠᵃ(i, j, k, grid, c) = @inbounds c[i, j,   k] - c[i, j-1, k]

@inline δzᵃᵃᶜ(i, j, k, grid, w) = @inbounds w[i, j, k+1] - w[i, j,   k]
@inline δzᵃᵃᶠ(i, j, k, grid, c) = @inbounds c[i, j,   k] - c[i, j, k-1]

#####
##### Difference operators acting on functions
#####

@inline δxᶜᵃᵃ(i, j, k, grid, f::F, args...) where F<:Function = f(i+1, j, k, grid, args...) - f(i,   j, k, grid, args...)
@inline δxᶠᵃᵃ(i, j, k, grid, f::F, args...) where F<:Function = f(i,   j, k, grid, args...) - f(i-1, j, k, grid, args...)

@inline δyᵃᶜᵃ(i, j, k, grid, f::F, args...) where F<:Function = f(i, j+1, k, grid, args...) - f(i, j,   k, grid, args...)
@inline δyᵃᶠᵃ(i, j, k, grid, f::F, args...) where F<:Function = f(i, j,   k, grid, args...) - f(i, j-1, k, grid, args...)

@inline δzᵃᵃᶜ(i, j, k, grid, f::F, args...) where F<:Function = f(i, j, k+1, grid, args...) - f(i, j, k,   grid, args...)
@inline δzᵃᵃᶠ(i, j, k, grid, f::F, args...) where F<:Function = f(i, j, k,   grid, args...) - f(i, j, k-1, grid, args...)

#####
##### Three dimensional, homogeneous differences
#####

const XFlatGrid = AG{<:Any, Flat}
const YFlatGrid = AG{<:Any, <:Any, Flat}
const ZFlatGrid = AG{<:Any, <:Any, <:Any, Flat}

const c = Center()
const f = Face()

syms = (:ᶜ, :ᶠ)
locs = (:c, :f)

for (sx, ℓx) in zip(syms, locs)
    for (sy, ℓy) in zip(syms, locs)
        for (sz, ℓz) in zip(syms, locs)

            δx_inner = Symbol(:δx, sx, :ᵃ, :ᵃ)
            δy_inner = Symbol(:δy, :ᵃ, sy, :ᵃ)
            δz_inner = Symbol(:δz, :ᵃ, :ᵃ, sz)

            for (ξ, δin) in zip((:x, :y, :z), (δx_inner, δy_inner, δz_inner))
                boundary = Symbol(ξ, :_boundary)
                δout = Symbol(:δ, ξ, sx, sy, sz)

                @eval begin
                    @inline $δout(i, j, k, grid, q) =
                        ifelse($boundary(i, j, k, grid, $ℓx, $ℓy, $ℓz), zero(grid), $δin(i, j, k, grid, q))

                    @inline $δout(i, j, k, grid, q::F, args...) where {F<:Function} =
                        ifelse($boundary(i, j, k, grid, $ℓx, $ℓy, $ℓz), zero(grid), $δin(i, j, k, grid, q, args...))
                end
            end

            # Support for Flat Earths

            δx_outer = Symbol(:δx, sx, sy, sz)
            δy_outer = Symbol(:δy, sx, sy, sz)
            δz_outer = Symbol(:δz, sx, sy, sz)

            @eval begin
                @inline $δx_outer(i, j, k, grid::XFlatGrid, u) = zero(grid)
                @inline $δx_outer(i, j, k, grid::XFlatGrid, f::F, args...) where F<:Function = zero(grid)

                @inline $δy_outer(i, j, k, grid::YFlatGrid, u) = zero(grid)
                @inline $δy_outer(i, j, k, grid::YFlatGrid, f::F, args...) where F<:Function = zero(grid)

                @inline $δz_outer(i, j, k, grid::ZFlatGrid, u) = zero(grid)
                @inline $δz_outer(i, j, k, grid::ZFlatGrid, f::F, args...) where F<:Function = zero(grid)
            end
        end
    end
end

