
struct HybridOrderWENO{NH, NL, FT, A1, A2} <: AbstractUpwindBiasedAdvectionScheme{NH, FT}
   high_order_scheme :: A1
    low_order_scheme :: A2

    HybridOrderWENO{NH, NL, FT}(ho::A1, lo::A2) where {NH, NL, FT, A1, A2} = 
        new{NH, NL, FT, A1, A2}(ho, lo)
end

HybridOrderWENO(grid, FT::DataType=Float64; kwargs...) = HybridOrderWENO(FT; grid, kwargs...)

function HybridOrderWENO(FT::DataType=Float64; 
                         grid = nothing,
                         high_order = 9,
                         low_order  = 5) 

    high_order_scheme = WENO(grid, FT; order = high_order)
     low_order_scheme = WENO(grid, FT; order = low_order)
    
    NH = boundary_buffer(high_order_scheme)
    NL = boundary_buffer(low_order_scheme)
    
    return HybridOrderWENO{NH, NL, FT}(high_order_scheme,
                                        low_order_scheme)
end

Adapt.adapt_structure(to, scheme::HybridOrderWENO{NH, NL, FT}) where {NH, NL, FT} =
    HybridOrderWENO{NH, NL, FT}(Adapt.adapt(to, scheme.high_order_scheme), 
                                Adapt.adapt(to, scheme.low_order_scheme))

for (bias, stencil) in zip((:left, :right), (0, -1))
    for (dir, loc) in zip((:x, :x, :y, :y, :z, :z), (:ᶠᵃᵃ, :ᶜᵃᵃ, :ᵃᶠᵃ, :ᵃᶜᵃ, :ᵃᵃᶠ, :ᵃᵃᶜ))
        alt_interp = Symbol(:_, bias, :_biased_interpolate_, dir, loc)
        biased_β   = Symbol(bias, :_biased_β)
        
        @eval begin
            function $alt_interp(i, j, k, grid, scheme::HybridOrderWENO{5, 3}, f::Function, args...)

                rᴴ = $alt_interp(i, j, k, grid, scheme.high_order_scheme, f, args...)
                rᴸ = $alt_interp(i, j, k, grid, scheme.low_order_scheme,  f, args...)

                Sᴴ = $(reconstruction_stencil(5, bias, dir, true))
                Sᴸ = $(reconstruction_stencil(3, bias, dir, true))

                βᴴ = $biased_β(scheme.high_order_scheme, Sᴴ, Val($stencil))
                βᴸ = $biased_β(scheme.low_order_scheme,  Sᴸ, Val($stencil)) 

                μᴴ = ((βᴴ + FT(ε))^ƞ)
                μᴸ = ((βᴸ + FT(ε))^ƞ)

                αᴴ = ifelse(μᴴ >= μᴸ, μᴸ / μᴴ, 1)
                αᴸ = 1 - αᴴ

                return (rᴴ * αᴴ + rᴸ * αᴸ) 
            end

            function $alt_interp(i, j, k, grid, scheme::HybridOrderWENO{5, 3}, f) 

                rᴴ = $alt_interp(i, j, k, grid, scheme.high_order_scheme, f)
                rᴸ = $alt_interp(i, j, k, grid, scheme.low_order_scheme,  f)

                Sᴴ = $(reconstruction_stencil(5, bias, dir, false))
                Sᴸ = $(reconstruction_stencil(3, bias, dir, false))

                βᴴ = $biased_β(scheme.high_order_scheme, Sᴴ, Val($stencil))
                βᴸ = $biased_β(scheme.low_order_scheme,  Sᴸ, Val($stencil)) 

                μᴴ = ((βᴴ + FT(ε))^ƞ)
                μᴸ = ((βᴸ + FT(ε))^ƞ)

                αᴴ = ifelse(μᴴ >= μᴸ, μᴸ / μᴴ, 1)
                αᴸ = 1 - αᴴ

                return (rᴴ * αᴴ + rᴸ * αᴸ) 
            end
        end
    end
end
                