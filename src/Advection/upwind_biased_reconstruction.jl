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

    # Enforce the grid type if a grid is provided
    FT = grid isa Nothing ? FT : eltype(grid) 
    
    mod(order, 2) == 0 && throw(ArgumentError("UpwindBiased reconstruction scheme is defined only for odd orders"))

    N = Int((order + 1) ÷ 2)
    symmetric_order = ifelse(N > 1, order-1, 2)
    advecting_velocity_scheme = Centered(FT; order = symmetric_order)

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
    for (F, bool) in zip((:Any, :Callable), (false, true))
        for FT in fully_supported_float_types
            interp = Symbol(:biased_interpolate_, dir, side)
            @eval begin
                @inline $interp(i, j, k, grid, ::UpwindBiased{1, $FT}, red_order::Int, bias, ψ::$F, args...) = ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 1, :left,  dir, bool)), 
                                                                                                                                         $(stencil_reconstruction(FT, 1, :right, dir, bool)))

                @inline function $interp(i, j, k, grid, ::UpwindBiased{2, $FT}, red_order::Int, bias, ψ::$F, args...)          
                    ifelse(red_order==1,
                           ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 1, :left,  dir, bool)), 
                                                     $(stencil_reconstruction(FT, 1, :right, dir, bool))),
                           ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 2, :left,  dir, bool)), 
                                                     $(stencil_reconstruction(FT, 2, :right, dir, bool))))
                end

                @inline function $interp(i, j, k, grid, ::UpwindBiased{3, $FT}, red_order::Int, bias, ψ::$F, args...)          
                    ifelse(red_order==1,
                           ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 1, :left,  dir, bool)), 
                                                     $(stencil_reconstruction(FT, 1, :right, dir, bool))),
                    ifelse(red_order==2,
                           ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 2, :left,  dir, bool)), 
                                                     $(stencil_reconstruction(FT, 2, :right, dir, bool))),
                           ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 3, :left,  dir, bool)), 
                                                     $(stencil_reconstruction(FT, 3, :right, dir, bool)))))
                end

                @inline function $interp(i, j, k, grid, ::UpwindBiased{4, $FT}, red_order::Int, bias, ψ::$F, args...)          
                    ifelse(red_order==1,
                           ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 1, :left,  dir, bool)), 
                                                     $(stencil_reconstruction(FT, 1, :right, dir, bool))),
                    ifelse(red_order==2,
                           ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 2, :left,  dir, bool)), 
                                                     $(stencil_reconstruction(FT, 2, :right, dir, bool))),
                    ifelse(red_order==3,
                           ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 3, :left,  dir, bool)), 
                                                     $(stencil_reconstruction(FT, 3, :right, dir, bool))),
                           ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 4, :left,  dir, bool)), 
                                                     $(stencil_reconstruction(FT, 4, :right, dir, bool))))))
                end

                @inline function $interp(i, j, k, grid, ::UpwindBiased{5, $FT}, red_order::Int, bias, ψ::$F, args...)          
                    ifelse(red_order==1,
                           ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 1, :left,  dir, bool)), 
                                                     $(stencil_reconstruction(FT, 1, :right, dir, bool))),
                    ifelse(red_order==2,
                           ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 2, :left,  dir, bool)), 
                                                     $(stencil_reconstruction(FT, 2, :right, dir, bool))),
                    ifelse(red_order==3,
                           ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 3, :left,  dir, bool)), 
                                                     $(stencil_reconstruction(FT, 3, :right, dir, bool))),
                    ifelse(red_order==4,
                           ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 4, :left,  dir, bool)), 
                                                     $(stencil_reconstruction(FT, 4, :right, dir, bool))),
                           ifelse(bias isa LeftBias, $(stencil_reconstruction(FT, 5, :left,  dir, bool)), 
                                                     $(stencil_reconstruction(FT, 5, :right, dir, bool)))))))
                end
            end
        end
    end
end
