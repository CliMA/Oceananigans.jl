#####
##### High-order reconstructions reduce their order where their stencil would read beyond the indices the sides of the
##### domain let it read or, on immersed grids, an inactive cell.
#####

using Oceananigans.Grids: AbstractGrid, BoundarySide, TwoSided, Closed, Open, topology

const AG = AbstractGrid

# Topologies with a Closed or an Open side
const BT = Union{TwoSided{<:BoundarySide}, TwoSided{<:Any, <:BoundarySide}}

const AGX = AG{<:Any, <:BT}
const AGY = AG{<:Any, <:Any, <:BT}
const AGZ = AG{<:Any, <:Any, <:Any, <:BT}

# Indices a stencil of points at `ℓ` may read next to an Open side.
@inline first_stencil_index(::Closed, ℓ)      = 1
@inline first_stencil_index(::Open, ::Center) = 0
@inline first_stencil_index(::Open, ::Face)   = 1

@inline last_stencil_index(::Closed, ::Center, N) = N
@inline last_stencil_index(::Closed, ::Face, N)   = N + 1
@inline last_stencil_index(::Open, ℓ, N)          = N + 1

# Whether a stencil from `first_point` to `last_point` reads only what the sides of a direction let it read; only Closed and Open sides bound it
@inline within_bounds(first_point, last_point, topology, ℓ, N) = true
@inline within_bounds(first_point, last_point, ::Type{<:TwoSided{L, R}}, ℓ, N) where {L <: BoundarySide, R <: BoundarySide} = (first_point >= first_stencil_index(L(), ℓ)) & (last_point <= last_stencil_index(R(), ℓ, N))
@inline within_bounds(first_point, last_point, ::Type{<:TwoSided{L}}, ℓ, N) where L <: BoundarySide         = first_point >= first_stencil_index(L(), ℓ)
@inline within_bounds(first_point, last_point, ::Type{<:TwoSided{<:Any, R}}, ℓ, N) where R <: BoundarySide  = last_point  <= last_stencil_index(R(), ℓ, N)

# A reconstruction at a Face reads Center points from i - B to i + B - 1 and one at a Center reads Face points from i - B + 1
# to i + B, with B the buffer of the scheme; a left-biased stencil drops the last point and a right-biased one the first.
for (d, ξ) in enumerate((:x, :y, :z))
    N = (:Nx, :Ny, :Nz)[d]
    index = (:i, :j, :k)[d]
    B = Symbol(:required_halo_size_, ξ)

    for (side, points, first_offset, last_offset) in ((:ᶠ, Center(), :(-$B(scheme)),    :($B(scheme) - 1)),
                                                      (:ᶜ, Face(),   :(1 - $B(scheme)), :($B(scheme))))

        within_bounds_symmetric = Symbol(:within_bounds_symmetric_, ξ, side)
        within_bounds_biased    = Symbol(:within_bounds_biased_, ξ, side)
        near_boundary_symmetric = Symbol(:near_, ξ, :_boundary_symmetric, side)
        near_boundary_biased    = Symbol(:near_, ξ, :_boundary_biased, side)

        @eval begin
            @inline $within_bounds_symmetric(i, grid, scheme) =
                within_bounds(i + $first_offset, i + $last_offset, topology(grid, $d), $points, grid.$N)

            @inline function $within_bounds_biased(i, grid, scheme, bias)
                iᴸ = i + $first_offset + ifelse(bias == LeftBias, 0, 1)
                iᴿ = i + $last_offset  - ifelse(bias == LeftBias, 1, 0)
                return within_bounds(iᴸ, iᴿ, topology(grid, $d), $points, grid.$N)
            end

            @inline $near_boundary_symmetric(i, j, k, grid, scheme)     = !$within_bounds_symmetric($index, grid, scheme)
            @inline $near_boundary_biased(i, j, k, grid, scheme, bias) = !$within_bounds_biased($index, grid, scheme, bias)
        end
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
        bounded_grid = (:AGX, :AGY, :AGZ)[d]
        b = bias == :biased ? (:bias,) : ()

        for loc in (:ᶜ, :ᶠ), (alt1, alt2) in zip((:_, :__, :___, :____, :_____), (:_____, :_, :__, :___, :____))
            code[d] = loc
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            alt1_interp = Symbol(alt1, interp)
            alt2_interp = Symbol(alt2, interp)
            near_boundary = Symbol(:near_, ξ, :_boundary_, bias, loc)

            @eval begin
                # Directions without a Closed or Open side, on grids without an immersed boundary, and low-order schemes
                @inline $alt1_interp(i, j, k, grid::AG, scheme::HOADV, args...) = $interp(i, j, k, grid, scheme, args...)
                @inline $alt1_interp(i, j, k, grid::AG, scheme::LOADV, args...) = $interp(i, j, k, grid, scheme, args...)

                # On immersed grids the condition changes from cell to cell, so both reconstructions are evaluated
                @inline $alt1_interp(i, j, k, grid::ImmersedBoundaryGrid, scheme::HOADV, $(b...), args...) =
                    ifelse($near_boundary(i, j, k, grid, scheme, $(b...)),
                           $alt2_interp(i, j, k, grid, scheme.buffer_scheme, $(b...), args...),
                           $interp(i, j, k, grid, scheme, $(b...), args...))
            end

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
            if bias == :biased
                @eval @inline function $alt1_interp(i, j, k, grid::$bounded_grid, scheme::HOADV, bias, args...)
                    ψ̂ = $interp(i, j, k, grid, scheme, bias, args...)
                    if $near_boundary(i, j, k, grid, scheme, bias)
                        # BFloat16 WENO reconstructions are Float32, the lowest-order buffer schemes BFloat16
                        return convert(typeof(ψ̂), $alt2_interp(i, j, k, grid, scheme.buffer_scheme, bias, args...))
                    else
                        return ψ̂
                    end
                end
            else
                @eval @inline $alt1_interp(i, j, k, grid::$bounded_grid, scheme::HOADV, args...) =
                    ifelse($near_boundary(i, j, k, grid, scheme),
                           $alt2_interp(i, j, k, grid, scheme.buffer_scheme, args...),
                           $interp(i, j, k, grid, scheme, args...))
            end
        end
    end
end
