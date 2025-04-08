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

for dir in (:x, :y, :z)
    outside_symmetric_haloᶠ = Symbol(:outside_symmetric_halo_, dir, :ᶠ)
    outside_symmetric_haloᶜ = Symbol(:outside_symmetric_halo_, dir, :ᶜ)
    outside_biased_haloᶠ = Symbol(:outside_biased_halo_, dir, :ᶠ)
    outside_biased_haloᶜ = Symbol(:outside_biased_halo_, dir, :ᶜ)
    required_halo_size = Symbol(:required_halo_size_, dir)

    @eval begin
     @inline $outside_symmetric_haloᶠ(i, N, adv) = (i >= $required_halo_size(adv) + 1) & (i <= N + 1 - $required_halo_size(adv))
     @inline $outside_symmetric_haloᶜ(i, N, adv) = (i >= $required_halo_size(adv))     & (i <= N + 1 - $required_halo_size(adv))

     @inline $outside_biased_haloᶠ(i, N, adv) = (i >= $required_halo_size(adv) + 1) & (i <= N + 1 - ($required_halo_size(adv) - 1)) &  # Left bias
                                                (i >= $required_halo_size(adv))     & (i <= N + 1 - $required_halo_size(adv))          # Right bias
     @inline $outside_biased_haloᶜ(i, N, adv) = (i >= $required_halo_size(adv))     & (i <= N + 1 - ($required_halo_size(adv) - 1)) &  # Left bias
                                                (i >= $required_halo_size(adv) - 1) & (i <= N + 1 - $required_halo_size(adv))          # Right bias
    end
end
# Separate High order advection from low order advection
const HOADV = Union{WENO, 
                    Tuple(Centered{N} for N in advection_buffers[2:end])...,
                    Tuple(UpwindBiased{N} for N in advection_buffers[2:end])...} 
const LOADV = Union{UpwindBiased{1}, Centered{1}}

for bias in (:symmetric, :biased)
    for (d, ξ) in enumerate((:x, :y, :z))

        code = [:ᵃ, :ᵃ, :ᵃ]

        for loc in (:ᶜ, :ᶠ), (alt1, alt2) in zip((:_, :__, :___, :____, :_____), (:_____, :_, :__, :___, :____))
            code[d] = loc
            second_order_interp = Symbol(:ℑ, ξ, code...)
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            alt1_interp = Symbol(alt1, interp)
            alt2_interp = Symbol(alt2, interp)

            # Simple translation for Periodic directions and low-order advection schemes (fallback)
            @eval @inline $alt1_interp(i, j, k, grid::AUG, scheme::HOADV, args...) = $interp(i, j, k, grid, scheme, args...)
            @eval @inline $alt1_interp(i, j, k, grid::AUG, scheme::LOADV, args...) = $interp(i, j, k, grid, scheme, args...)

            # Disambiguation
            for GridType in [:AUGX, :AUGY, :AUGZ, :AUGXY, :AUGXZ, :AUGYZ, :AUGXYZ]
                @eval @inline $alt1_interp(i, j, k, grid::$GridType, scheme::LOADV, args...) = $interp(i, j, k, grid, scheme, args...)
            end

            outside_buffer = Symbol(:outside_, bias, :_halo_, ξ, loc)

            # Conditional high-order interpolation in Bounded directions
            if ξ == :x
                @eval begin
                    @inline $alt1_interp(i, j, k, grid::AUGX, scheme::HOADV, args...) =
                        ifelse($outside_buffer(i, grid.Nx, scheme),
                               $interp(i, j, k, grid, scheme, args...),
                               $alt2_interp(i, j, k, grid, scheme.buffer_scheme, args...))
                end
            elseif ξ == :y
                @eval begin
                    @inline $alt1_interp(i, j, k, grid::AUGY, scheme::HOADV, args...) =
                        ifelse($outside_buffer(j, grid.Ny, scheme),
                               $interp(i, j, k, grid, scheme, args...),
                               $alt2_interp(i, j, k, grid, scheme.buffer_scheme, args...))
                end
            elseif ξ == :z
                @eval begin
                    @inline $alt1_interp(i, j, k, grid::AUGZ, scheme::HOADV, args...) =
                        ifelse($outside_buffer(k, grid.Nz, scheme),
                               $interp(i, j, k, grid, scheme, args...),
                               $alt2_interp(i, j, k, grid, scheme.buffer_scheme, args...))
                end
            end
        end
    end
end

@inline _multi_dimensional_reconstruction_x(i, j, k, grid::AUGX, scheme, interp, args...) = 
                    ifelse(outside_symmetric_bufferᶜ(i, grid.Nx, scheme), 
                           multi_dimensional_reconstruction_x(i, j, k, grid::AUGX, scheme, interp, args...),
                           interp(i, j, k, grid, scheme, args...))

@inline _multi_dimensional_reconstruction_y(i, j, k, grid::AUGY, scheme, interp, args...) = 
                    ifelse(outside_symmetric_bufferᶜ(j, grid.Ny, scheme), 
                            multi_dimensional_reconstruction_y(i, j, k, grid::AUGY, scheme, interp, args...),
                            interp(i, j, k, grid, scheme, args...))
