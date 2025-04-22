#####
##### Centered advection scheme
#####

"""
    struct Centered{N, FT, XT, YT, ZT, CA} <: AbstractCenteredAdvectionScheme{N, FT}

Centered reconstruction scheme.
"""
struct Centered{N, FT} <: AbstractCenteredAdvectionScheme{N, FT} end

function Centered(FT::DataType=Oceananigans.defaults.FloatType; grid = nothing, order = 2) 

    if !(grid isa Nothing) 
        FT = eltype(grid)
    end

    mod(order, 2) != 0 && throw(ArgumentError("Centered reconstruction scheme is defined only for even orders"))

    N = Int(order ÷ 2)
    return Centered{N, FT}()
end

Base.summary(a::Centered{N}) where N = string("Centered(order=", 2N, ")")
Base.show(io::IO, a::Centered{N, FT}) where {N, FT} = summary(a)

# Useful aliases
Centered(grid, FT::DataType=Float64; kwargs...) = Centered(FT; grid, kwargs...)

const ACAS = AbstractCenteredAdvectionScheme

# left and right biased for Centered reconstruction are just symmetric!
@inline _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::ACAS, bias, c, args...) = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c, args...)
@inline _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::ACAS, bias, c, args...) = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c, args...)
@inline _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::ACAS, bias, c, args...) = _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c, args...)
@inline _biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::ACAS, bias, c, args...) = _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, c, args...)
@inline _biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::ACAS, bias, c, args...) = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, c, args...)
@inline _biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::ACAS, bias, c, args...) = _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, c, args...)

for (side, dir) in zip((:ᶠᵃᵃ, :ᵃᶠᵃ, :ᵃᵃᶠ), (:x, :y, :z))
    for (F, bool) in zip((:Any, :(Base.Callable)), (false, true))
        for FT in fully_supported_float_types
            interp = Symbol(:symmetric_interpolate_, dir, side)
            @eval begin
                @inline $interp(i, j, k, grid, ::Centered{1, $FT}, red_order::Int, ψ::$F, args...) = $(calc_reconstruction_stencil(FT, 1, :symmetric, dir, bool))

                @inline function $interp(i, j, k, grid, ::Centered{2, $FT}, red_order::Int, ψ::$F, args...)          
                    ifelse(red_order==1, 
                           $(calc_reconstruction_stencil(FT, 1, :symmetric, dir, bool)),
                           $(calc_reconstruction_stencil(FT, 2, :symmetric, dir, bool)))
                end

                @inline function $interp(i, j, k, grid, ::Centered{3, $FT}, red_order::Int, ψ::$F, args...)          
                    ifelse(red_order==1,
                           $(calc_reconstruction_stencil(FT, 1, :symmetric, dir, bool)),
                    ifelse(red_order==2,
                           $(calc_reconstruction_stencil(FT, 2, :symmetric, dir, bool)),
                           $(calc_reconstruction_stencil(FT, 3, :symmetric, dir, bool))))
                end

                @inline function $interp(i, j, k, grid, ::Centered{4, $FT}, red_order::Int, ψ::$F, args...)          
                    ifelse(red_order==1,
                        $(calc_reconstruction_stencil(FT, 1, :symmetric, dir, bool)),
                    ifelse(red_order==2,
                        $(calc_reconstruction_stencil(FT, 2, :symmetric, dir, bool)),
                    ifelse(red_order==3,
                        $(calc_reconstruction_stencil(FT, 3, :symmetric, dir, bool)),
                        $(calc_reconstruction_stencil(FT, 4, :symmetric, dir, bool)))))
                end

                @inline function $interp(i, j, k, grid, ::Centered{5, $FT}, red_order::Int, ψ::$F, args...)          
                    ifelse(red_order==1,
                           $(calc_reconstruction_stencil(FT, 1, :symmetric, dir, bool)),
                    ifelse(red_order==2,
                           $(calc_reconstruction_stencil(FT, 2, :symmetric, dir, bool)),
                    ifelse(red_order==3,
                           $(calc_reconstruction_stencil(FT, 3, :symmetric, dir, bool)),
                    ifelse(red_order==4,
                           $(calc_reconstruction_stencil(FT, 4, :symmetric, dir, bool)),
                           $(calc_reconstruction_stencil(FT, 5, :symmetric, dir, bool))))))
                end

                @inline function $interp(i, j, k, grid, ::Centered{6, $FT}, red_order::Int, ψ::$F, args...)          
                    ifelse(red_order==1,
                           $(calc_reconstruction_stencil(FT, 1, :symmetric, dir, bool)),
                    ifelse(red_order==2,
                           $(calc_reconstruction_stencil(FT, 2, :symmetric, dir, bool)),
                    ifelse(red_order==3,
                           $(calc_reconstruction_stencil(FT, 3, :symmetric, dir, bool)),
                    ifelse(red_order==4,
                           $(calc_reconstruction_stencil(FT, 4, :symmetric, dir, bool)),
                    ifelse(red_order==5,
                           $(calc_reconstruction_stencil(FT, 5, :symmetric, dir, bool)),
                           $(calc_reconstruction_stencil(FT, 6, :symmetric, dir, bool)))))))
                end
            end
        end
    end
end