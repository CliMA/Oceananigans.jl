
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

weno_weights(β, ::Val{5}, FT) = β[1]^ƞ + β[2]^ƞ + β[3]^ƞ + β[4]^ƞ + β[5]^ƞ + FT(ε)^ƞ
weno_weights(β, ::Val{3}, FT) = β[1]^ƞ + β[2]^ƞ + β[3]^ƞ + FT(ε)^ƞ

for bias in (:left, :right)
    for (dir, loc) in zip((:x, :x, :y, :y, :z, :z), (:ᶠᵃᵃ, :ᶜᵃᵃ, :ᵃᶠᵃ, :ᵃᶜᵃ, :ᵃᵃᶠ, :ᵃᵃᶜ))
        alt_interp = Symbol(:_, bias, :_biased_interpolate_, dir, loc)
        biased_β   = Symbol(bias, :_biased_β)
        stencil    = Symbol(bias, :_stencil_, dir, loc)

        @eval begin
            function $alt_interp(i, j, k, grid, scheme::HybridOrderWENO{5, 3, FT}, ψ, args...) where FT

                rᴴ = $alt_interp(i, j, k, grid, scheme.high_order_scheme, ψ, args...)
                rᴸ = $alt_interp(i, j, k, grid, scheme.low_order_scheme,  ψ, args...)

                Sᴴ = $stencil(i, j, k, scheme.high_order_scheme, ψ, grid, args...)
                Sᴸ = $stencil(i, j, k, scheme.low_order_scheme,  ψ, grid, args...)
                
                βᴴ = beta_loop(scheme.high_order_scheme, Sᴴ, $biased_β)
                βᴸ = beta_loop(scheme.low_order_scheme,  Sᴸ, $biased_β)

                αᴸ = weno_weights(βᴴ, Val(5), FT) / 5
                αᴴ = weno_weights(βᴸ, Val(3), FT) / 3

                ∑α = αᴸ + αᴴ

                return (rᴴ * αᴴ + rᴸ * αᴸ) / ∑α
            end

            function $alt_interp(i, j, k, grid, scheme::HybridOrderWENO{5, 3, FT}, ψ, VI::AbstractSmoothnessStencil, args...) where FT

                rᴴ = $alt_interp(i, j, k, grid, scheme.high_order_scheme, ψ, VI, args...)
                rᴸ = $alt_interp(i, j, k, grid, scheme.low_order_scheme,  ψ, VI, args...)

                Sᴴ = $stencil(i, j, k, scheme.high_order_scheme, ψ, grid, args...)
                Sᴸ = $stencil(i, j, k, scheme.low_order_scheme,  ψ, grid, args...)
                
                βᴴ = beta_loop(scheme.high_order_scheme, Sᴴ, $biased_β)
                βᴸ = beta_loop(scheme.low_order_scheme,  Sᴸ, $biased_β)

                αᴸ = weno_weights(βᴴ, Val(5), FT) / 5
                αᴴ = weno_weights(βᴸ, Val(3), FT) / 3

                ∑α = αᴸ + αᴴ

                return (rᴴ * αᴴ + rᴸ * αᴸ) / ∑α
            end
        end
    end
end
                