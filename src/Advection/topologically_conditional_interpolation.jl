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

using Oceananigans.Grids: AbstractGrid,
                          Bounded,
                          RightConnected,
                          LeftConnected,
                          topology,
                          architecture

const AG = AbstractGrid

# topologies bounded at least on one side
const BT = Union{Bounded, RightConnected, LeftConnected}

# Bounded underlying Grids
const AGX   = AG{<:Any, <:BT}
const AGY   = AG{<:Any, <:Any, <:BT}
const AGZ   = AG{<:Any, <:Any, <:Any, <:BT}
const AGXY  = AG{<:Any, <:BT, <:BT}
const AGXZ  = AG{<:Any, <:BT, <:Any, <:BT}
const AGYZ  = AG{<:Any, <:Any, <:BT, <:BT}
const AGXYZ = AG{<:Any, <:BT, <:BT, <:BT}

# Left-biased buffers are smaller by one grid point on the right side; vice versa for right-biased buffers
# Center interpolation stencil look at i + 1 (i.e., require one less point on the left)

for dir in (:x, :y, :z)
    outside_symmetric_haloᶠ = Symbol(:outside_symmetric_halo_, dir, :ᶠ)
    outside_symmetric_haloᶜ = Symbol(:outside_symmetric_halo_, dir, :ᶜ)
    outside_biased_haloᶠ    = Symbol(:outside_biased_halo_, dir, :ᶠ)
    outside_biased_haloᶜ    = Symbol(:outside_biased_halo_, dir, :ᶜ)
    required_halo_size      = Symbol(:required_halo_size_, dir)

    @eval begin
        # Bounded topologies
        @inline $outside_symmetric_haloᶠ(i, ::Type{Bounded}, N, adv) = (i >= $required_halo_size(adv) + 1) & (i <= N + 1 - $required_halo_size(adv))
        @inline $outside_symmetric_haloᶜ(i, ::Type{Bounded}, N, adv) = (i >= $required_halo_size(adv))     & (i <= N + 1 - $required_halo_size(adv))

        @inline $outside_biased_haloᶠ(i, ::Type{Bounded}, N, adv) = (i >= $required_halo_size(adv) + 1) & (i <= N + 1 - ($required_halo_size(adv) - 1)) &  # Left bias
                                                                    (i >= $required_halo_size(adv))     & (i <= N + 1 - $required_halo_size(adv))          # Right bias
        @inline $outside_biased_haloᶜ(i, ::Type{Bounded}, N, adv) = (i >= $required_halo_size(adv))     & (i <= N + 1 - ($required_halo_size(adv) - 1)) &  # Left bias
                                                                    (i >= $required_halo_size(adv) - 1) & (i <= N + 1 - $required_halo_size(adv))          # Right bias

        # Right connected topologies (only test the left side, i.e. the bounded side)
        @inline $outside_symmetric_haloᶠ(i, ::Type{RightConnected}, N, adv) = i >= $required_halo_size(adv) + 1
        @inline $outside_symmetric_haloᶜ(i, ::Type{RightConnected}, N, adv) = i >= $required_halo_size(adv)

        @inline $outside_biased_haloᶠ(i, ::Type{RightConnected}, N, adv) = (i >= $required_halo_size(adv) + 1) &  # Left bias
                                                                           (i >= $required_halo_size(adv))        # Right bias
        @inline $outside_biased_haloᶜ(i, ::Type{RightConnected}, N, adv) = (i >= $required_halo_size(adv))     &  # Left bias
                                                                           (i >= $required_halo_size(adv) - 1)    # Right bias

        # Left bounded topologies (only test the right side, i.e. the bounded side)
        @inline $outside_symmetric_haloᶠ(i, ::Type{LeftConnected}, N, adv) = (i <= N + 1 - $required_halo_size(adv))
        @inline $outside_symmetric_haloᶜ(i, ::Type{LeftConnected}, N, adv) = (i <= N + 1 - $required_halo_size(adv))

        @inline $outside_biased_haloᶠ(i, ::Type{LeftConnected}, N, adv) = (i <= N + 1 - ($required_halo_size(adv) - 1)) &  # Left bias
                                                                          (i <= N + 1 - $required_halo_size(adv))          # Right bias
        @inline $outside_biased_haloᶜ(i, ::Type{LeftConnected}, N, adv) = (i <= N + 1 - ($required_halo_size(adv) - 1)) &  # Left bias
                                                                          (i <= N + 1 - $required_halo_size(adv))          # Right bias
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
            @eval @inline $alt1_interp(i, j, k, grid::AG, scheme::HOADV, args...) = $interp(i, j, k, grid, scheme, args...)
            @eval @inline $alt1_interp(i, j, k, grid::AG, scheme::LOADV, args...) = $interp(i, j, k, grid, scheme, args...)

            outside_buffer = Symbol(:outside_, bias, :_halo_, ξ, loc)

            # Conditional high-order interpolation in Bounded directions
            if ξ == :x
                @eval begin
                    @inline $alt1_interp(i, j, k, grid::AGX, scheme::HOADV, args...) =
                            ifelse($outside_buffer(i, topology(grid, 1), grid.Nx, scheme),
                                   $interp(i, j, k, grid, scheme, args...),
                                   $alt2_interp(i, j, k, grid, scheme.buffer_scheme, args...))
                end
            elseif ξ == :y
                @eval begin
                    @inline $alt1_interp(i, j, k, grid::AGY, scheme::HOADV, args...) =
                        ifelse($outside_buffer(j, topology(grid, 2), grid.Ny, scheme),
                               $interp(i, j, k, grid, scheme, args...),
                               $alt2_interp(i, j, k, grid, scheme.buffer_scheme, args...))
                end
            elseif ξ == :z
                @eval begin
                    @inline $alt1_interp(i, j, k, grid::AGZ, scheme::HOADV, args...) =
                        ifelse($outside_buffer(k, topology(grid, 3), grid.Nz, scheme),
                               $interp(i, j, k, grid, scheme, args...),
                               $alt2_interp(i, j, k, grid, scheme.buffer_scheme, args...))
                end
            end
        end
    end
end

@inline _multi_dimensional_reconstruction_x(i, j, k, grid::AGX, scheme, interp, args...) =
                    ifelse(outside_symmetric_bufferᶜ(i, topology(grid, 1), grid.Nx, scheme),
                           multi_dimensional_reconstruction_x(i, j, k, grid, scheme, interp, args...),
                           interp(i, j, k, grid, scheme, args...))

@inline _multi_dimensional_reconstruction_y(i, j, k, grid::AGY, scheme, interp, args...) =
                    ifelse(outside_symmetric_bufferᶜ(j, topology(grid, 2), grid.Ny, scheme),
                            multi_dimensional_reconstruction_y(i, j, k, grid, scheme, interp, args...),
                            interp(i, j, k, grid, scheme, args...))
