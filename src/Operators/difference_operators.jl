using Oceananigans.Grids: Flat, immersed_inactive_node, topology, unit_vector, within_bounds

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
##### Support for Flat Earths
#####

@inline δxᶜᵃᵃ(i, j, k, grid::AG{FT, Flat, TY, TZ}, u) where {FT, TY, TZ} = zero(FT)
@inline δxᶠᵃᵃ(i, j, k, grid::AG{FT, Flat, TY, TZ}, c) where {FT, TY, TZ} = zero(FT)

@inline δyᵃᶜᵃ(i, j, k, grid::AG{FT, TX, Flat, TZ}, v) where {FT, TX, TZ} = zero(FT)
@inline δyᵃᶠᵃ(i, j, k, grid::AG{FT, TX, Flat, TZ}, c) where {FT, TX, TZ} = zero(FT)

@inline δzᵃᵃᶜ(i, j, k, grid::AG{FT, TX, TY, Flat}, w) where {FT, TX, TY} = zero(FT)
@inline δzᵃᵃᶠ(i, j, k, grid::AG{FT, TX, TY, Flat}, c) where {FT, TX, TY} = zero(FT)

@inline δxᶜᵃᵃ(i, j, k, grid::AG{FT, Flat, TY, TZ}, f::F, args...) where {FT, TY, TZ, F<:Function} = zero(FT)
@inline δxᶠᵃᵃ(i, j, k, grid::AG{FT, Flat, TY, TZ}, f::F, args...) where {FT, TY, TZ, F<:Function} = zero(FT)

@inline δyᵃᶜᵃ(i, j, k, grid::AG{FT, TX, Flat, TZ}, f::F, args...) where {FT, TX, TZ, F<:Function} = zero(FT)
@inline δyᵃᶠᵃ(i, j, k, grid::AG{FT, TX, Flat, TZ}, f::F, args...) where {FT, TX, TZ, F<:Function} = zero(FT)

@inline δzᵃᵃᶜ(i, j, k, grid::AG{FT, TX, TY, Flat}, f::F, args...) where {FT, TX, TY, F<:Function} = zero(FT)
@inline δzᵃᵃᶠ(i, j, k, grid::AG{FT, TX, TY, Flat}, f::F, args...) where {FT, TX, TY, F<:Function} = zero(FT)

#####
##### 3D and 2D differences
#####

# Export one dimensional difference operators
export δxᶠᵃᵃ, δxᶜᵃᵃ
export δyᵃᶠᵃ, δyᵃᶜᵃ
export δzᵃᵃᶠ, δzᵃᵃᶜ

#####
##### A 3D difference is zero when one of its operands has no data: beyond the sides of the differenced direction, or
##### inside an immersed boundary
#####

# The operands, along d, of a difference or an interpolation to (ℓx, ℓy, ℓz): the nodes n - 1 and n of a Face, n and n + 1
# of a Center, at the location flipped along d
@inline function operands(i, j, k, ::Val{d}, ℓx, ℓy, ℓz) where d
    di, dj, dk = unit_vector(Val(d))
    s = ifelse((ℓx, ℓy, ℓz)[d] isa Face, -1, 0)
    n⁻ = (i + s * di, j + s * dj, k + s * dk)
    n⁺ = (n⁻[1] + di, n⁻[2] + dj, n⁻[3] + dk)
    ℓᵒ = (d == 1 ? flip_location(ℓx) : ℓx, d == 2 ? flip_location(ℓy) : ℓy, d == 3 ? flip_location(ℓz) : ℓz)
    return n⁻, n⁺, ℓᵒ
end

@inline function conditional_δ(i, j, k, grid, ::Val{d}, ℓx, ℓy, ℓz, δ, args...) where d
    n⁻, n⁺, ℓᵒ = operands(i, j, k, Val(d), ℓx, ℓy, ℓz)
    other = (ℓx, ℓy, ℓz)[d] isa Face ? n⁻ : n⁺
    no_data = !within_bounds(n⁻[d], n⁺[d], topology(grid, d), ℓᵒ[d], size(grid, d)) |
              immersed_inactive_node(i, j, k, grid, ℓᵒ...) |
              immersed_inactive_node(other..., grid, ℓᵒ...)
    return ifelse(no_data, zero(grid), δ(i, j, k, grid, args...))
end

# Define and export 2D and 3D differences
for ℓ1 in (:ᶜ, :ᶠ), ℓ2 in (:ᶜ, :ᶠ, :ᵃ), ℓ3 in (:ᶜ, :ᶠ, :ᵃ)
    if !(ℓ2 == ℓ3 == :ᵃ) # 1D differences are defined above!
        δx = Symbol(:δx, ℓ1, ℓ2, ℓ3)
        δy = Symbol(:δy, ℓ2, ℓ1, ℓ3)
        δz = Symbol(:δz, ℓ2, ℓ3, ℓ1)

        δxᵃ = Symbol(:δx, ℓ1, :ᵃ, :ᵃ)
        δyᵃ = Symbol(:δy, :ᵃ, ℓ1, :ᵃ)
        δzᵃ = Symbol(:δz, :ᵃ, :ᵃ, ℓ1)

        if ℓ2 == :ᵃ || ℓ3 == :ᵃ
            @eval begin
                @inline $δx(i, j, k, grid, f::Function, args...) = $δxᵃ(i, j, k, grid, f, args...)
                @inline $δy(i, j, k, grid, f::Function, args...) = $δyᵃ(i, j, k, grid, f, args...)
                @inline $δz(i, j, k, grid, f::Function, args...) = $δzᵃ(i, j, k, grid, f, args...)

                @inline $δx(i, j, k, grid, c) = $δxᵃ(i, j, k, grid, c)
                @inline $δy(i, j, k, grid, c) = $δyᵃ(i, j, k, grid, c)
                @inline $δz(i, j, k, grid, c) = $δzᵃ(i, j, k, grid, c)
            end
        else
            ℓ = ℓ1 == :ᶜ ? :(Center()) : :(Face())
            ℓa = ℓ2 == :ᶜ ? :(Center()) : :(Face())
            ℓb = ℓ3 == :ᶜ ? :(Center()) : :(Face())

            @eval begin
                @inline $δx(i, j, k, grid, f::Function, args...) = conditional_δ(i, j, k, grid, Val(1), $ℓ, $ℓa, $ℓb, $δxᵃ, f, args...)
                @inline $δy(i, j, k, grid, f::Function, args...) = conditional_δ(i, j, k, grid, Val(2), $ℓa, $ℓ, $ℓb, $δyᵃ, f, args...)
                @inline $δz(i, j, k, grid, f::Function, args...) = conditional_δ(i, j, k, grid, Val(3), $ℓa, $ℓb, $ℓ, $δzᵃ, f, args...)

                @inline $δx(i, j, k, grid, c) = conditional_δ(i, j, k, grid, Val(1), $ℓ, $ℓa, $ℓb, $δxᵃ, c)
                @inline $δy(i, j, k, grid, c) = conditional_δ(i, j, k, grid, Val(2), $ℓa, $ℓ, $ℓb, $δyᵃ, c)
                @inline $δz(i, j, k, grid, c) = conditional_δ(i, j, k, grid, Val(3), $ℓa, $ℓb, $ℓ, $δzᵃ, c)
            end
        end

        @eval export $δx, $δy, $δz
    end
end
