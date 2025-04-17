#####
##### Upwind-biased 3rd-order advection scheme
#####

"""
    struct UpwindBiased <: AbstractUpwindBiasedAdvectionScheme{3}

Upwind-biased reconstruction scheme.
"""
struct UpwindBiased{N, FT, SI} <: AbstractUpwindBiasedAdvectionScheme{N, FT} 
    "Reconstruction scheme used for symmetric interpolation"
    advecting_velocity_scheme :: SI

    UpwindBiased{N, FT}(advecting_velocity_scheme::SI) where {N, FT, SI} = new{N, FT, SI}(advecting_velocity_scheme)
end

function UpwindBiased(FT::DataType = Float64; grid = nothing, order = 3) 

    if !(grid isa Nothing) 
        FT = eltype(grid)
    end

    mod(order, 2) == 0 && throw(ArgumentError("UpwindBiased reconstruction scheme is defined only for odd orders"))

    N  = Int((order + 1) ÷ 2)

    if N > 1
        advecting_velocity_scheme = Centered(FT; grid, order = order - 1)
    else
        advecting_velocity_scheme = Centered(FT; grid, order = 2)
    end

    return UpwindBiased{N, FT}(advecting_velocity_scheme)
end

Base.summary(a::UpwindBiased{N}) where N = string("UpwindBiased(order=", 2N-1, ")")

Base.show(io::IO, a::UpwindBiased{N, FT}) where {N, FT} =
    print(io, summary(a), " \n",
              " Symmetric scheme: ", "\n",
              "    └── ", summary(a.advecting_velocity_scheme))

Adapt.adapt_structure(to, scheme::UpwindBiased{N, FT}) where {N, FT} =
    UpwindBiased{N, FT}(Adapt.adapt(to, scheme.advecting_velocity_scheme))

on_architecture(to, scheme::UpwindBiased{N, FT}) where {N, FT} =
    UpwindBiased{N, FT}(on_architecture(to, scheme.advecting_velocity_scheme))

# Useful aliases
UpwindBiased(grid, FT::DataType=Float64; kwargs...) = UpwindBiased(FT; grid, kwargs...)

const AUAS = AbstractUpwindBiasedAdvectionScheme

# symmetric interpolation for UpwindBiased and WENO
@inline _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::AUAS, args...) = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::AUAS, args...) = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::AUAS, args...) = _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::AUAS, args...) = _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::AUAS, args...) = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::AUAS, args...) = _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)

for (side, dir) in zip((:ᶠᵃᵃ, :ᵃᶠᵃ, :ᵃᵃᶠ), (:x, :y, :z))
    for (F, bool) in zip((:Any, :(Base.Callable)), (false, true))
        for FT in fully_supported_float_types
            interp = Symbol(:biased_interpolate_, dir, side)
            @eval begin
                @inline $interp(i, j, k, grid, ::UpwindBiased{1, $FT}, red_order::Int, bias, ψ::$F, args...) = ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 1, :left,  dir, bool)), 
                                                                                                                                         $(calc_reconstruction_stencil(FT, 1, :right, dir, bool)))

                @inline function $interp(i, j, k, grid, ::UpwindBiased{2, $FT}, red_order::Int, bias, ψ::$F, args...)          
                    if red_order==1
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 1, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 1, :right, dir, bool)))
                    else
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 2, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 2, :right, dir, bool)))
                    end
                end

                @inline function $interp(i, j, k, grid, ::UpwindBiased{3, $FT}, red_order::Int, bias, ψ::$F, args...)          
                    if red_order==1
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 1, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 1, :right, dir, bool)))
                    elseif red_order==2
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 2, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 2, :right, dir, bool)))
                    else
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 3, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 3, :right, dir, bool)))
                    end
                end

                @inline function $interp(i, j, k, grid, ::UpwindBiased{4, $FT}, red_order::Int, bias, ψ::$F, args...)          
                    if red_order==1
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 1, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 1, :right, dir, bool)))
                    elseif red_order==2
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 2, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 2, :right, dir, bool)))
                    elseif red_order==3
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 3, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 3, :right, dir, bool)))
                    else
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 4, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 4, :right, dir, bool)))
                    end
                end

                @inline function $interp(i, j, k, grid, ::UpwindBiased{5, $FT}, red_order::Int, bias, ψ::$F, args...)          
                    if red_order==1
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 1, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 1, :right, dir, bool)))
                    elseif red_order==2
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 2, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 2, :right, dir, bool)))
                    elseif red_order==3
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 3, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 3, :right, dir, bool)))
                    elseif red_order==4
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 4, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 4, :right, dir, bool)))
                    else
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 5, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 5, :right, dir, bool)))
                    end
                end

                @inline function $interp(i, j, k, grid, ::UpwindBiased{6, $FT}, red_order::Int, bias, ψ::$F, args...)          
                    if red_order==1
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 1, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 1, :right, dir, bool)))
                    elseif red_order==2
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 2, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 2, :right, dir, bool)))
                    elseif red_order==3
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 3, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 3, :right, dir, bool)))
                    elseif red_order==4
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 4, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 4, :right, dir, bool)))
                    elseif red_order==5
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 5, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 5, :right, dir, bool)))
                    else
                        ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, 6, :left,  dir, bool)), 
                                                  $(calc_reconstruction_stencil(FT, 6, :right, dir, bool)))
                    end
                end
            end
        end
    end
end

# Uniform upwind biased reconstruction
for buffer in advection_buffers, FT in fully_supported_float_types
    @eval begin
        # Flat fluxes...
        @inline biased_interpolate_xᶠᵃᵃ(i, j, k, grid::XFlatGrid, ::UpwindBiased{$buffer, $FT}, ::Int, bias, ψ, args...) = @inbounds ψ[i, j, k]
        @inline biased_interpolate_xᶠᵃᵃ(i, j, k, grid::XFlatGrid, ::UpwindBiased{$buffer, $FT}, ::Int, bias, ψ::Base.Callable, args...) = ψ(i, j, k, grid, args...)
        @inline biased_interpolate_yᵃᶠᵃ(i, j, k, grid::YFlatGrid, ::UpwindBiased{$buffer, $FT}, ::Int, bias, ψ, args...) = @inbounds ψ[i, j, k]
        @inline biased_interpolate_yᵃᶠᵃ(i, j, k, grid::YFlatGrid, ::UpwindBiased{$buffer, $FT}, ::Int, bias, ψ::Base.Callable, args...) = ψ(i, j, k, grid, args...)
        @inline biased_interpolate_zᵃᵃᶠ(i, j, k, grid::ZFlatGrid, ::UpwindBiased{$buffer, $FT}, ::Int, bias, ψ, args...) = @inbounds ψ[i, j, k]
        @inline biased_interpolate_zᵃᵃᶠ(i, j, k, grid::ZFlatGrid, ::UpwindBiased{$buffer, $FT}, ::Int, bias, ψ::Base.Callable, args...) = ψ(i, j, k, grid, args...)
    end
end