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

@inline    outside_symmetric_bufferᶠ(i, N, adv) = (i >= boundary_buffer(adv) + 1) & (i <= N + 1 - boundary_buffer(adv))
@inline    outside_symmetric_bufferᶜ(i, N, adv) = (i >= boundary_buffer(adv))     & (i <= N + 1 - boundary_buffer(adv))
@inline  outside_left_biased_bufferᶠ(i, N, adv) = (i >= boundary_buffer(adv) + 1) & (i <= N + 1 - (boundary_buffer(adv) - 1))
@inline  outside_left_biased_bufferᶜ(i, N, adv) = (i >= boundary_buffer(adv))     & (i <= N + 1 - (boundary_buffer(adv) - 1))
@inline outside_right_biased_bufferᶠ(i, N, adv) = (i >= boundary_buffer(adv))     & (i <= N + 1 - boundary_buffer(adv))
@inline outside_right_biased_bufferᶜ(i, N, adv) = (i >= boundary_buffer(adv) - 1) & (i <= N + 1 - boundary_buffer(adv))

# Separate High order advection from low order advection
const HOADV = Union{Tuple(WENO{N} for N in advection_buffers[2:end])..., 
                    Tuple(Centered{N} for N in advection_buffers[2:end])...,
                    Tuple(UpwindBiased{N} for N in advection_buffers[2:end])...} 
const LOADV = Union{WENO{1}, UpwindBiased{1}, Centered{1}}

# Simple translation for Periodic directions and low-order advection schemes (fallback)
@inline _topologically_conditional_scheme_x(i, j, k, ::AUG, u, l, scheme::LOADV) = scheme
@inline _topologically_conditional_scheme_x(i, j, k, ::AUG, u, l, scheme::HOADV) = scheme
@inline _topologically_conditional_scheme_y(i, j, k, ::AUG, u, l, scheme::LOADV) = scheme
@inline _topologically_conditional_scheme_y(i, j, k, ::AUG, u, l, scheme::HOADV) = scheme
@inline _topologically_conditional_scheme_z(i, j, k, ::AUG, u, l, scheme::LOADV) = scheme
@inline _topologically_conditional_scheme_z(i, j, k, ::AUG, u, l, scheme::HOADV) = scheme

# Disambiguation
for GridType in [:AUGX, :AUGY, :AUGZ, :AUGXY, :AUGXZ, :AUGYZ, :AUGXYZ]
    @eval begin
        @inline _topologically_conditional_scheme_x(i, j, k, ::$GridType, u, l, scheme::LOADV) = scheme
        @inline _topologically_conditional_scheme_y(i, j, k, ::$GridType, u, l, scheme::LOADV) = scheme
        @inline _topologically_conditional_scheme_z(i, j, k, ::$GridType, u, l, scheme::LOADV) = scheme
    end
end

bias_identifyier(::Val{:LeftBiasedStencil})  = :left_biased
bias_identifyier(::Val{:RightBiasedStencil}) = :right_biased
bias_identifyier(::Val{:SymmetricStencil})   = :symmetric

for Dir in (:SymmetricStencil, :LeftBiasedStencil, :RightBiasedStencil), Loc in (:Face, :Center)
    loc  = Loc == :Face ? Symbol("ᶠ") : Symbol("ᶜ")
    bias = bias_identifyier(Val(Dir))
    outside_buffer = Symbol(:outside_, bias, :_buffer, loc)

    @eval begin
        # Conditional high-order interpolation in Bounded directions
        @inline _topologically_conditional_scheme_x(i, j, k, grid::AUGX, dir::$Dir, l::Type{$Loc}, scheme::HOADV) =
                ifelse($outside_buffer(i, grid.Nx, scheme), scheme,
                   _topologically_conditional_scheme_x(i, j, k, grid, dir, l, scheme.buffer_scheme))

        @inline _topologically_conditional_scheme_y(i, j, k, grid::AUGY, dir::$Dir, l::Type{$Loc}, scheme::HOADV) =
                ifelse($outside_buffer(j, grid.Ny, scheme), scheme,
                    _topologically_conditional_scheme_y(i, j, k, grid, dir, l, scheme.buffer_scheme))

        @inline _topologically_conditional_scheme_z(i, j, k, grid::AUGZ, dir::$Dir, l::Type{$Loc}, scheme::HOADV) =
                ifelse($outside_buffer(j, grid.Ny, scheme), scheme,
                    _topologically_conditional_scheme_y(i, j, k, grid, dir, l, scheme.buffer_scheme))
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
