
struct HybridOrderWENO{NH, NM, NL, FT, A1, A2, A3} <: AbstractUpwindBiasedAdvectionScheme{NH, FT}
   high_order_scheme :: A1
    mid_order_scheme :: A2
    low_order_scheme :: A3

    HybridOrderWENO{NH, NM, NL, FT}(ho::A1, mo::A2, lo::A3) where {NH, NM, NL, FT, A1, A2, A3} = 
        new{NH, NM, NL, FT, A1, A2, A3}(ho, mo, lo)
end

HybridOrderWENO(grid, FT::DataType=Float64; kwargs...) = HybridOrderWENO(FT; grid, kwargs...)

function HybridOrderWENO(FT::DataType=Float64; 
                         grid = nothing,
                         high_order = 9,
                         mid_order  = 7,
                         low_order  = 5) 

    high_order_scheme = WENO(grid, FT; order = high_order)
    mid_order_scheme  = WENO(grid, FT; order = mid_order)
    low_order_scheme  = WENO(grid, FT; order = low_order)
    
    NH = boundary_buffer(high_order_scheme)
    NM = boundary_buffer(mid_order_scheme)
    NL = boundary_buffer(low_order_scheme)
    
    return HybridOrderWENO{NH, NM, NL, FT}(high_order_scheme,
                                            mid_order_scheme,
                                            low_order_scheme)
end

Adapt.adapt_structure(to, scheme::HybridOrderWENO{NH, NM, NL, FT}) where {NH, NM, NL, FT} =
    HybridOrderWENO{NH, NM, NL, FT}(Adapt.adapt(to, scheme.high_order_scheme), 
                                    Adapt.adapt(to, scheme.mid_order_scheme), 
                                    Adapt.adapt(to, scheme.low_order_scheme))


left_stencil_xᶠᵃᵃ(args...) = left_stencil_x(args...)
left_stencil_yᵃᶠᵃ(args...) = left_stencil_y(args...)
left_stencil_zᵃᵃᶠ(args...) = left_stencil_z(args...)

right_stencil_xᶠᵃᵃ(args...) = right_stencil_x(args...)
right_stencil_yᵃᶠᵃ(args...) = right_stencil_y(args...)
right_stencil_zᵃᵃᶠ(args...) = right_stencil_z(args...)

left_stencil_xᶜᵃᵃ(i, j, k, args...) = left_stencil_x(i+1, j, k, args...)
left_stencil_yᵃᶜᵃ(i, j, k, args...) = left_stencil_y(i, j+1, k, args...)
left_stencil_zᵃᵃᶜ(i, j, k, args...) = left_stencil_z(i, j, k+1, args...)

right_stencil_xᶜᵃᵃ(i, j, k, args...) = right_stencil_x(i+1, j, k, args...)
right_stencil_yᵃᶜᵃ(i, j, k, args...) = right_stencil_y(i, j+1, k, args...)
right_stencil_zᵃᵃᶜ(i, j, k, args...) = right_stencil_z(i, j, k+1, args...)

scaling_weights(β, FT) = sum(β.^ƞ) + FT(ε)^ƞ

_symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::HybridOrderWENO, args...) = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.low_order_scheme, args...)
_symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::HybridOrderWENO, args...) = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.low_order_scheme, args...)
_symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::HybridOrderWENO, args...) = _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.low_order_scheme, args...)

_symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::HybridOrderWENO, args...) = _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.low_order_scheme, args...)
_symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::HybridOrderWENO, args...) = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.low_order_scheme, args...)
_symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::HybridOrderWENO, args...) = _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme.low_order_scheme, args...)

for bias in (:left, :right)
    for (dir, loc) in zip((:x, :x, :y, :y, :z, :z), (:ᶠᵃᵃ, :ᶜᵃᵃ, :ᵃᶠᵃ, :ᵃᶜᵃ, :ᵃᵃᶠ, :ᵃᵃᶜ))
        alt_interp = Symbol(:_, bias, :_biased_interpolate_, dir, loc)
        biased_β   = Symbol(bias, :_biased_β)
        stencil    = Symbol(bias, :_stencil_, dir, loc)

        @eval begin
            function $alt_interp(i, j, k, grid, scheme::HybridOrderWENO{NH, NM, NL, FT}, ψ, args...) where {NH, NM, NL, FT}

                rᴴ = $alt_interp(i, j, k, grid, scheme.high_order_scheme, ψ, args...)
                rᴹ = $alt_interp(i, j, k, grid, scheme.mid_order_scheme,  ψ, args...)
                rᴸ = $alt_interp(i, j, k, grid, scheme.low_order_scheme,  ψ, args...)

                Sᴴ = $stencil(i, j, k, scheme.high_order_scheme, ψ, grid, args...)
                Sᴹ = $stencil(i, j, k, scheme.mid_order_scheme,  ψ, grid, args...)
                Sᴸ = $stencil(i, j, k, scheme.low_order_scheme,  ψ, grid, args...)
                
                βᴴ = beta_loop(scheme.high_order_scheme, Sᴴ, $biased_β)
                βᴹ = beta_loop(scheme.mid_order_scheme,  Sᴹ, $biased_β)
                βᴸ = beta_loop(scheme.low_order_scheme,  Sᴸ, $biased_β)

                αᴴ = NH / scaling_weights(βᴴ, FT)
                αᴹ = NM / scaling_weights(βᴹ, FT)
                αᴸ = NL / scaling_weights(βᴸ, FT)

                ∑α = αᴴ + αᴹ + αᴸ 

                return (αᴴ * rᴴ + αᴹ * rᴹ + αᴸ * rᴸ) / ∑α
            end

            function $alt_interp(i, j, k, grid, scheme::HybridOrderWENO{NH, NM, NL, FT}, ψ, VI::AbstractSmoothnessStencil, args...) where {NH, NM, NL, FT}

                rᴴ = $alt_interp(i, j, k, grid, scheme.high_order_scheme, ψ, VI, args...)
                rᴹ = $alt_interp(i, j, k, grid, scheme.mid_order_scheme,  ψ, VI, args...)
                rᴸ = $alt_interp(i, j, k, grid, scheme.low_order_scheme,  ψ, VI, args...)

                Sᴴ = $stencil(i, j, k, scheme.high_order_scheme, ψ, grid, args...)
                Sᴹ = $stencil(i, j, k, scheme.mid_order_scheme,  ψ, grid, args...)
                Sᴸ = $stencil(i, j, k, scheme.low_order_scheme,  ψ, grid, args...)
                
                βᴴ = beta_loop(scheme.high_order_scheme, Sᴴ, $biased_β)
                βᴹ = beta_loop(scheme.mid_order_scheme,  Sᴹ, $biased_β)
                βᴸ = beta_loop(scheme.low_order_scheme,  Sᴸ, $biased_β)

                αᴴ = NH / scaling_weights(βᴴ, FT)
                αᴹ = NM / scaling_weights(βᴹ, FT)
                αᴸ = NL / scaling_weights(βᴸ, FT)

                ∑α = αᴴ + αᴹ + αᴸ 

                return (αᴴ * rᴴ + αᴹ * rᴹ + αᴸ * rᴸ) / ∑α
            end
        end
    end
end
                