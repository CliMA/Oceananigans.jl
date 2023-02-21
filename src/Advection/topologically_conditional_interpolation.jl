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

# Bounded underlying Grids
const AUGX   = AUG{<:Any, <:Bounded}
const AUGY   = AUG{<:Any, <:Any, <:Bounded}
const AUGZ   = AUG{<:Any, <:Any, <:Any, <:Bounded}
const AUGXY  = AUG{<:Any, <:Bounded, <:Bounded}
const AUGXZ  = AUG{<:Any, <:Bounded, <:Any, <:Bounded}
const AUGYZ  = AUG{<:Any, <:Any, <:Bounded, <:Bounded}
const AUGXYZ = AUG{<:Any, <:Bounded, <:Bounded, <:Bounded}

# Left-biased buffers are smaller by one grid point on the right side; vice versa for right-biased buffers
# Center interpolation stencil look at i + 1 (i.e., require one less point on the left)

@inline outside_symmetric_bufferᶠ(i, N, adv) = (i >= boundary_buffer(adv) + 1) & (i <= N + 1 - boundary_buffer(adv))
@inline outside_symmetric_bufferᶜ(i, N, adv) = (i >= boundary_buffer(adv))     & (i <= N + 1 - boundary_buffer(adv))

@inline outside_biased_bufferᶠ(i, N, adv, ::Val{:left})  = (i >= boundary_buffer(adv) + 1) & (i <= N + 1 - (boundary_buffer(adv) - 1))
@inline outside_biased_bufferᶜ(i, N, adv, ::Val{:left})  = (i >= boundary_buffer(adv))     & (i <= N + 1 - (boundary_buffer(adv) - 1))
@inline outside_biased_bufferᶠ(i, N, adv, ::Val{:right}) = (i >= boundary_buffer(adv))     & (i <= N + 1 - boundary_buffer(adv))
@inline outside_biased_bufferᶜ(i, N, adv, ::Val{:right}) = (i >= boundary_buffer(adv) - 1) & (i <= N + 1 - boundary_buffer(adv))

# Separate High order advection from low order advection
const HOADV = Union{WENO, Centered, UpwindBiased} 
const LOADV = Union{VectorInvariant, UpwindBiased{1}, Centered{1}}

for (d, ξ) in enumerate((:x, :y, :z))

    code = [:ᵃ, :ᵃ, :ᵃ]

    for loc in (:ᶜ, :ᶠ)
        code[d] = loc
        second_order_interp = Symbol(:ℑ, ξ, code...)
        interp = Symbol(:symmetric_interpolate_, ξ, code...)
        alt_interp = Symbol(:_, interp)

        # Simple translation for Periodic directions and low-order advection schemes (fallback)
        @eval $alt_interp(i, j, k, grid::AUG, scheme::LOADV, args...) = $interp(i, j, k, grid, scheme, args...)
        @eval $alt_interp(i, j, k, grid::AUG, scheme::HOADV, args...) = $interp(i, j, k, grid, scheme, args...)

        # Disambiguation
        for GridType in [:AUGX, :AUGY, :AUGZ, :AUGXY, :AUGXZ, :AUGYZ, :AUGXYZ]
            @eval $alt_interp(i, j, k, grid::$GridType, scheme::LOADV, args...) = $interp(i, j, k, grid, scheme, args...)
        end

        outside_buffer = Symbol(:outside_symmetric_buffer, loc)

        # Conditional high-order interpolation in Bounded directions
        if ξ == :x
            @eval begin
                @inline $alt_interp(i, j, k, grid::AUGX, scheme::HOADV, ψ) =
                    ifelse($outside_buffer(i, grid.Nx, scheme),
                            $interp(i, j, k, grid, scheme, ψ),
                            $alt_interp(i, j, k, grid, scheme.buffer_scheme, ψ))

                @inline $alt_interp(i, j, k, grid::AUGX, scheme::WENO, ζ, VI::AbstractSmoothnessStencil, u, v) =
                    ifelse($outside_buffer(i, grid.Nx, scheme),
                            $interp(i, j, k, grid, scheme, ζ, VI, u, v),
                            $alt_interp(i, j, k, grid, scheme.buffer_scheme, ζ, VI, u, v))
            end
        elseif ξ == :y
            @eval begin
                @inline $alt_interp(i, j, k, grid::AUGY, scheme::HOADV, ψ) =
                    ifelse($outside_buffer(j, grid.Ny, scheme),
                            $interp(i, j, k, grid, scheme, ψ),
                            $alt_interp(i, j, k, grid, scheme.buffer_scheme, ψ))

                @inline $alt_interp(i, j, k, grid::AUGY, scheme::WENO, ζ, VI::AbstractSmoothnessStencil, u, v) =
                    ifelse($outside_buffer(j, grid.Ny, scheme),
                            $interp(i, j, k, grid, scheme, ζ, VI, u, v),
                            $alt_interp(i, j, k, grid, scheme.buffer_scheme, ζ, VI, u, v))
            end
        elseif ξ == :z
            @eval begin
                @inline $alt_interp(i, j, k, grid::AUGZ, scheme::HOADV, ψ) =
                    ifelse($outside_buffer(k, grid.Nz, scheme),
                            $interp(i, j, k, grid, scheme, ψ),
                            $alt_interp(i, j, k, grid, scheme.buffer_scheme, ψ))
            end
        end
    end
end

for (d, ξ) in enumerate((:x, :y, :z))

    code = [:ᵃ, :ᵃ, :ᵃ]

    for loc in (:ᶜ, :ᶠ)
        code[d] = loc
        second_order_interp = Symbol(:ℑ, ξ, code...)
        interp = Symbol(:biased_interpolate_, ξ, code...)
        alt_interp = Symbol(:_, interp)

        # Simple translation for Periodic directions and low-order advection schemes (fallback)
        @eval $alt_interp(i, j, k, grid::AUG, scheme::LOADV, ::Val{D}, args...) where D = $interp(i, j, k, grid, scheme, Val(D), args...)
        @eval $alt_interp(i, j, k, grid::AUG, scheme::HOADV, ::Val{D}, args...) where D = $interp(i, j, k, grid, scheme, Val(D), args...)

        # Disambiguation
        for GridType in [:AUGX, :AUGY, :AUGZ, :AUGXY, :AUGXZ, :AUGYZ, :AUGXYZ]
            @eval $alt_interp(i, j, k, grid::$GridType, scheme::LOADV, ::Val{D}, args...) where D = $interp(i, j, k, grid, scheme, Val(D), args...)
        end

        outside_buffer = Symbol(:outside_biased_buffer, loc)

        # Conditional high-order interpolation in Bounded directions
        if ξ == :x
            @eval begin
                @inline $alt_interp(i, j, k, grid::AUGX, scheme::HOADV, ::Val{D}, ψ) where D =
                    ifelse($outside_buffer(i, grid.Nx, scheme, Val(D)),
                            $interp(i, j, k, grid, scheme, Val(D), ψ),
                            $alt_interp(i, j, k, grid, scheme.buffer_scheme, Val(D), ψ))

                @inline $alt_interp(i, j, k, grid::AUGX, scheme::WENO, ::Val{D}, ζ, VI::AbstractSmoothnessStencil, u, v) where D =
                    ifelse($outside_buffer(i, grid.Nx, scheme, Val(D)),
                            $interp(i, j, k, grid, scheme, Val(D), ζ, VI, u, v),
                            $alt_interp(i, j, k, grid, scheme.buffer_scheme, Val(D), ζ, VI, u, v))
            end
        elseif ξ == :y
            @eval begin
                @inline $alt_interp(i, j, k, grid::AUGY, scheme::HOADV, ::Val{D}, ψ) where D =
                    ifelse($outside_buffer(j, grid.Ny, scheme, Val(D)),
                            $interp(i, j, k, grid, scheme, Val(D), ψ),
                            $alt_interp(i, j, k, grid, scheme.buffer_scheme, Val(D), ψ))

                @inline $alt_interp(i, j, k, grid::AUGY, scheme::WENO, ::Val{D}, ζ, VI::AbstractSmoothnessStencil, u, v) where D =
                    ifelse($outside_buffer(j, grid.Ny, scheme, Val(D)),
                            $interp(i, j, k, grid, scheme, Val(D), ζ, VI, u, v),
                            $alt_interp(i, j, k, grid, scheme.buffer_scheme, Val(D), ζ, VI, u, v))
            end
        elseif ξ == :z
            @eval begin
                @inline $alt_interp(i, j, k, grid::AUGZ, scheme::HOADV, ::Val{D}, ψ) where D =
                    ifelse($outside_buffer(k, grid.Nz, scheme, Val(D)),
                            $interp(i, j, k, grid, scheme, Val(D), ψ),
                            $alt_interp(i, j, k, grid, scheme.buffer_scheme, Val(D), ψ))
            end
        end
    end
end
