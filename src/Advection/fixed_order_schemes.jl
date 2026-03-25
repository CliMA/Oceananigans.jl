using Oceananigans.Grids: AbstractGrid
using Oceananigans.ImmersedBoundaries

struct FixedOrderScheme{N, FT, S} <: AbstractAdvectionScheme{N, FT}
    scheme::S
    function FixedOrderScheme{N, FT}(scheme::S) where {N, FT}
        return new{N, FT, S}(scheme)
end

"""UpwindBiased advection scheme with fixed order"""
struct FixedOrderUBScheme{N, FT, S} <: AbstractUpwindBiasedAdvectionScheme{N, FT}
    scheme::S
    function FixedOrderUBScheme{N, FT}(scheme::S) where {N, FT}
        return new{N, FT, S}(scheme)
    end
end

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

function FixedOrderCentered(scheme::Centered{N, FT}) where {N, FT}
    return FixedOrderScheme{N, FT}(scheme)
end


function FixedOrderUpwindBiased(scheme::UpwindBiased{N, FT}) where {N, FT}
    return FixedOrderUBScheme{N, FT}(scheme)
end

function FixedOrderWENO(scheme::WENO{N, FT}) where {N, FT}
    return FixedOrderUBScheme{N, FT}(scheme)
end

# Overload all interpolation functions to skip any boundary checks
for bias in (:symmetric, :biased)
    for (d, ξ) in enumerate((:x, :y, :z))

        code = [:ᵃ, :ᵃ, :ᵃ]

        for loc in (:ᶜ, :ᶠ)
            code[d] = loc
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            _interp = Symbol(:_, interp)

            @eval begin
                @inline function $_interp(i, j, k, grid::AbstractGrid, scheme::FixedOrderScheme, args...)
                    return $interp(i, j, k, grid, scheme, args...)
                end
                @inline function $_interp(i, j, k, grid::AbstractGrid, scheme::FixedOrderUBScheme, args...)
                    return $interp(i, j, k, grid, scheme, args...)
                end
            end
        end
    end
end
