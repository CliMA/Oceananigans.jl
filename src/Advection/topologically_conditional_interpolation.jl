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
                          architecture,
                          LeftConnectedRightCenterConnected,
                          LeftConnectedRightCenterFolded,
                          LeftConnectedRightFaceConnected,
                          LeftConnectedRightFaceFolded

const AG = AbstractGrid

# topologies bounded at least on one side
const BT = Union{Bounded, RightConnected, LeftConnected,
                  LeftConnectedRightCenterFolded, LeftConnectedRightFaceFolded,
                  LeftConnectedRightCenterConnected, LeftConnectedRightFaceConnected}

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
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            alt1_interp = Symbol(alt1, interp)
            alt2_interp = Symbol(alt2, interp)

            # Simple translation for Periodic directions and low-order advection schemes (fallback)
            @eval @inline $alt1_interp(i, j, k, grid::AG, scheme::HOADV, args...) = $interp(i, j, k, grid, scheme, args...)
            @eval @inline $alt1_interp(i, j, k, grid::AG, scheme::LOADV, args...) = $interp(i, j, k, grid, scheme, args...)

            outside_buffer = Symbol(:outside_, bias, :_halo_, ξ, loc)

            # Conditional high-order interpolation in Bounded directions.
            #
            # Near a boundary the high-order stencil would reach outside the domain, so the interpolation falls back
            # on `scheme.buffer_scheme`, which in turn falls back on lower orders (e.g. WENO5, WENO3 and first-order
            # upwind beneath WENO7). `ifelse` evaluates both of its arguments, so with `ifelse` every point computes
            # this whole cascade of reduced-order reconstructions and then discards it. For upwind-biased (e.g. WENO)
            # reconstructions the cascade costs about as much as the high-order reconstruction itself, so we use an
            # `if` and evaluate the buffer schemes only where they are used.
            #
            # A branch in a GPU kernel is cheap only when all threads of a warp take the same side. The condition
            # depends only on the index along the bounded direction. In z it is uniform within every warp, because
            # Oceananigans launches GPU kernels with workgroups that span only x and y (e.g. 16 × 16), so all the
            # threads of a workgroup share the same k. In x and y it diverges only in the few warps that straddle a
            # boundary region.
            #
            # The high-order interpolation is evaluated before the branch, as with `ifelse`, so that its loads are
            # issued early (this is faster than evaluating it inside the branch). Symmetric interpolations are cheap
            # and keep `ifelse` (branching them is slower).
            index = (:i, :j, :k)[d]
            N     = (:Nx, :Ny, :Nz)[d]
            AGξ   = (:AGX, :AGY, :AGZ)[d]

            if bias == :biased
                @eval @inline function $alt1_interp(i, j, k, grid::$AGξ, scheme::HOADV, args...)
                    ψ̂ = $interp(i, j, k, grid, scheme, args...)
                    if $outside_buffer($index, topology(grid, $d), grid.$N, scheme)
                        return ψ̂
                    else
                        # The buffer schemes of a reduced-precision scheme may return a different floating-point type
                        # than the high-order reconstruction (e.g. BFloat16 versus Float32). Converting to the type of
                        # ψ̂ keeps the interpolation type-stable, rather than returning a `Union`.
                        return convert(typeof(ψ̂), $alt2_interp(i, j, k, grid, scheme.buffer_scheme, args...))
                    end
                end
            else
                @eval @inline $alt1_interp(i, j, k, grid::$AGξ, scheme::HOADV, args...) =
                    ifelse($outside_buffer($index, topology(grid, $d), grid.$N, scheme),
                           $interp(i, j, k, grid, scheme, args...),
                           $alt2_interp(i, j, k, grid, scheme.buffer_scheme, args...))
            end
        end
    end
end

@inline _multi_dimensional_reconstruction_x(i, j, k, grid::AGX, scheme, interp, args...) =
                    ifelse(outside_symmetric_halo_xᶜ(i, topology(grid, 1), grid.Nx, scheme),
                           multi_dimensional_reconstruction_x(i, j, k, grid, scheme, interp, args...),
                           interp(i, j, k, grid, scheme, args...))

@inline _multi_dimensional_reconstruction_y(i, j, k, grid::AGY, scheme, interp, args...) =
                    ifelse(outside_symmetric_halo_yᶜ(j, topology(grid, 2), grid.Ny, scheme),
                            multi_dimensional_reconstruction_y(i, j, k, grid, scheme, interp, args...),
                            interp(i, j, k, grid, scheme, args...))
