using Oceananigans.Grids: AbstractGrid,

fixed_order_scheme(scheme::Centered) = FixedOrderCentered(scheme)

fixed_order_scheme(scheme::UpwindBiased) = FixedOrderUpwindBiased(scheme)

fixed_order_scheme(scheme::WENO) = FixedOrderWENO(scheme)

function fixed_order_scheme(scheme::VectorInvariant)
    return VectorInvariant(
        vorticity_scheme=fixed_order_scheme(scheme.vorticity_scheme),
        vorticity_stencil=scheme.vorticity_stencil,
        vertical_advection_scheme=fixed_order_scheme(scheme.vertical_advection_scheme),
        divergence_scheme=fixed_order_scheme(scheme.divergence_scheme),
        kinetic_energy_gradient_scheme=fixed_order_scheme(scheme.kinetic_energy_gradient_scheme),
        upwinding=scheme.upwinding
    )
end

#Fallback, maybe should use a warning if not implemented?
fixed_order_scheme(scheme) = scheme

"""Centered advection scheme with fixed order"""
const FixedOrderCentered{N, FT} = Centered{N, FT, Nothing}

function FixedOrderCentered(scheme::Centered{N, FT}) where {N, FT}
    return Centered(FT; order=scheme_order(scheme), buffer_scheme=nothing)
end

"""UpwindBiased advection scheme with fixed order"""
const FixedOrderUpwindBiased{N, FT, SI} = UpwindBiased{N, FT, Nothing, SI}

function FixedOrderUpwindBiased(scheme::UpwindBiased{N, FT}) where {N, FT}
    return UpwindBiased(FT; order=scheme_order(scheme), buffer_scheme=nothing)
end

"""WENO advection scheme with fixed order"""
const FixedOrderWENO{N, FT, FT2, PP, SI} = WENO{N, FT, FT2, PP, Nothing, SI}

function FixedOrderWENO(FT::DataType=Oceananigans.defaults.FloatType, FT2::DataType=Float32;
              order = 5,
              bounds = nothing)
    return WENO(FT, FT2; order=order, buffer_scheme=nothing, bounds=bounds)
end

# Overload all interpolation functions to skip any boundary checks
for static_scheme in (:FixedOrderWENO, :FixedOrderCentered, :FixedOrderUpwindBiased)
  for bias in (:symmetric, :biased)
      for (d, ξ) in enumerate((:x, :y, :z))

          code = [:ᵃ, :ᵃ, :ᵃ]

          for loc in (:ᶜ, :ᶠ)
              code[d] = loc
              interp = Symbol(bias, :_interpolate_, ξ, code...)
              _interp = Symbol(:_, interp)

              @eval begin
                  @inline function $_interp(i, j, k, grid::AbstractGrid, scheme::$static_scheme, args...)
                      return $interp(i, j, k, grid, scheme, args...)
                  end
              end
          end
      end
  end
end
