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

@inline outside_symmetric_haloᶠ(i, N, adv) = (i >= required_halo_size(adv) + 1) & (i <= N + 1 - required_halo_size(adv))
@inline outside_symmetric_haloᶜ(i, N, adv) = (i >= required_halo_size(adv))     & (i <= N + 1 - required_halo_size(adv))

@inline  outside_left_biased_haloᶠ(i, N, adv) = (i >= required_halo_size(adv) + 1) & (i <= N + 1 - (required_halo_size(adv) - 1))
@inline  outside_left_biased_haloᶜ(i, N, adv) = (i >= required_halo_size(adv))     & (i <= N + 1 - (required_halo_size(adv) - 1))
@inline outside_right_biased_haloᶠ(i, N, adv) = (i >= required_halo_size(adv))     & (i <= N + 1 - required_halo_size(adv))
@inline outside_right_biased_haloᶜ(i, N, adv) = (i >= required_halo_size(adv) - 1) & (i <= N + 1 - required_halo_size(adv))

@inline function calculate_orderᶠ(i, N, ::Val{6}) 
    Oᴺ   = (i >= 7) & (i <= N - 5) 
    Oᴺ⁻¹ = (i >= 6) & (i <= N - 4)
    Oᴺ⁻² = (i >= 5) & (i <= N - 3)
    Oᴺ⁻³ = (i >= 4) & (i <= N - 2)
    Oᴺ⁻⁴ = (i >= 3) & (i <= N - 1)
    
    return 1 + Oᴺ + Oᴺ⁻¹ + Oᴺ⁻² + Oᴺ⁻³ + Oᴺ⁻⁴
end   

@inline function calculate_orderᶜ(i, N, ::Val{6}) 
    Oᴺ   = (i >= 6) & (i <= N - 5) 
    Oᴺ⁻¹ = (i >= 5) & (i <= N - 4)
    Oᴺ⁻² = (i >= 4) & (i <= N - 3)
    Oᴺ⁻³ = (i >= 3) & (i <= N - 2)
    Oᴺ⁻⁴ = (i >= 2) & (i <= N - 1)
    
    return 1 + Oᴺ + Oᴺ⁻¹ + Oᴺ⁻² + Oᴺ⁻³ + Oᴺ⁻⁴
end   

@inline function calculate_orderᶠ(i, N, ::Val{5}) 
    Oᴺ   = (i >= 6) & (i <= N - 4) 
    Oᴺ⁻¹ = (i >= 5) & (i <= N - 3)
    Oᴺ⁻² = (i >= 4) & (i <= N - 2)
    Oᴺ⁻³ = (i >= 3) & (i <= N - 1)
    
    return 1 + Oᴺ + Oᴺ⁻¹ + Oᴺ⁻² + Oᴺ⁻³
end   

@inline function calculate_orderᶜ(i, N, ::Val{5}) 
    Oᴺ   = (i >= 5) & (i <= N - 4) 
    Oᴺ⁻¹ = (i >= 4) & (i <= N - 3)
    Oᴺ⁻² = (i >= 3) & (i <= N - 2)
    Oᴺ⁻³ = (i >= 2) & (i <= N - 1)
    
    return 1 + Oᴺ + Oᴺ⁻¹ + Oᴺ⁻² + Oᴺ⁻³
end   

@inline function calculate_orderᶠ(i, N, ::Val{4}) 
    Oᴺ   = (i >= 5) & (i <= N - 3) 
    Oᴺ⁻¹ = (i >= 4) & (i <= N - 2)
    Oᴺ⁻² = (i >= 3) & (i <= N - 1)
    
    return 1 + Oᴺ + Oᴺ⁻¹ + Oᴺ⁻² 
end   

@inline function calculate_orderᶜ(i, N, ::Val{4}) 
    Oᴺ   = (i >= 4) & (i <= N - 3) 
    Oᴺ⁻¹ = (i >= 3) & (i <= N - 2)
    Oᴺ⁻² = (i >= 2) & (i <= N - 1)
    
    return 1 + Oᴺ + Oᴺ⁻¹ + Oᴺ⁻²
end   

@inline function calculate_orderᶠ(i, N, ::Val{3}) 
    Oᴺ   = (i >= 4) & (i <= N - 2) 
    Oᴺ⁻¹ = (i >= 3) & (i <= N - 1)
    
    return 1 + Oᴺ + Oᴺ⁻¹ 
end   

@inline function calculate_orderᶜ(i, N, ::Val{3}) 
    Oᴺ   = (i >= 3) & (i <= N - 2) 
    Oᴺ⁻¹ = (i >= 2) & (i <= N - 1)
    
    return 1 + Oᴺ + Oᴺ⁻¹ 
end   

@inline calculate_orderᶠ(i, N, ::Val{2}) = 1 + (i >= 3) & (i <= N - 1)
@inline calculate_orderᶜ(i, N, ::Val{2}) = 1 + (i >= 2) & (i <= N - 1) 

# Separate High order advection from low order advection
const HOADV = Union{WENO, 
                    Tuple(Centered{N} for N in advection_buffers[2:end])...,
                    Tuple(UpwindBiased{N} for N in advection_buffers[2:end])...} 
const LOADV = Union{UpwindBiased{1}, Centered{1}}

for bias in (:symmetric, :left_biased, :right_biased)
    for (d, ξ) in enumerate((:x, :y, :z))

        code = [:ᵃ, :ᵃ, :ᵃ]

        for loc in (:ᶜ, :ᶠ)
            code[d] = loc
            second_order_interp = Symbol(:ℑ, ξ, code...)
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            alt_interp = Symbol(:_, interp)
            calculate_order = Symbol(:calculate_order, loc)

            # Simple translation for Periodic directions and low-order advection schemes (fallback)
            @eval @inline $alt_interp(i, j, k, grid::AUG, schemxse::LOADV, ψ, args...) = $interp(i, j, k, grid, scheme, ψ, args...)
            @eval @inline $alt_interp(i, j, k, grid::AUG, scheme::HOADV,   ψ, args...) = $interp(i, j, k, grid, scheme, ψ, args...)
            if bias == :left_biased || bias == :right_biased
                @eval @inline $alt_interp(i, j, k, grid::AUG, scheme::WENO{N}, ψ, args...) where N = $interp(i, j, k, grid, scheme, ψ, N, args...)
            end

            # Disambiguation
            for GridType in [:AUGX, :AUGY, :AUGZ, :AUGXY, :AUGXZ, :AUGYZ, :AUGXYZ]
                @eval @inline $alt_interp(i, j, k, grid::$GridType, scheme::LOADV, args...) = $interp(i, j, k, grid, scheme, args...)
            end

            outside_buffer = Symbol(:outside_, bias, :_halo, loc)

            # Conditional high-order interpolation in Bounded directions
            if ξ == :x
                @eval begin
                    @inline $alt_interp(i, j, k, grid::AUGX, scheme::HOADV, args...) =
                        ifelse($outside_buffer(i, grid.Nx, scheme),
                               $interp(i, j, k, grid, scheme, args...),
                               $alt_interp(i, j, k, grid, scheme.buffer_scheme, args...))
                end

                if bias == :left_biased || bias == :right_biased
                    @eval begin 
                        @inline function $alt_interp(i, j, k, grid::AUGX, scheme::WENO{N}, ψ, args...) where N
                            order = $calculate_order(i, grid.Nx, Val(N))
                            return $interp(i, j, k, grid, scheme, ψ, order, args...)
                        end
                    end
                end
            elseif ξ == :y
                @eval begin
                    @inline $alt_interp(i, j, k, grid::AUGY, scheme::HOADV, args...) =
                        ifelse($outside_buffer(j, grid.Ny, scheme),
                               $interp(i, j, k, grid, scheme, args...),
                               $alt_interp(i, j, k, grid, scheme.buffer_scheme, args...))
                end

                if bias == :left_biased || bias == :right_biased
                    @eval begin 
                        @inline function $alt_interp(i, j, k, grid::AUGY, scheme::WENO{N}, ψ, args...) where N
                            order = $calculate_order(j, grid.Ny, Val(N))
                            return $interp(i, j, k, grid, scheme, ψ, order, args...)
                        end
                    end
                end
            elseif ξ == :z
                @eval begin
                    @inline $alt_interp(i, j, k, grid::AUGZ, scheme::HOADV, args...) =
                        ifelse($outside_buffer(k, grid.Nz, scheme),
                               $interp(i, j, k, grid, scheme, args...),
                               $alt_interp(i, j, k, grid, scheme.buffer_scheme, args...))
                end
                
                if bias == :left_biased || bias == :right_biased
                    @eval begin 
                        @inline function $alt_interp(i, j, k, grid::AUGZ, scheme::WENO{N}, ψ, args...) where N
                            order = $calculate_order(k, grid.Nz, Val(N))
                            return $interp(i, j, k, grid, scheme, ψ, order, args...)
                        end
                    end
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
