using Oceananigans.Grids: AbstractGrid
using Oceananigans.ImmersedBoundaries
using Oceananigans.Advection: Centered, UpwindBiased, WENO, VectorInvariant

struct CenteredFixedOrderAdvectionScheme{N, FT, S <: AbstractCenteredAdvectionScheme} <: AbstractCenteredAdvectionScheme{N, FT}
    scheme::S
    function CenteredFixedOrderAdvectionScheme{N, FT}(scheme::S) where {N, FT, S<:AbstractCenteredAdvectionScheme}
        return new{N, FT, S}(scheme)
    end
end

struct UpwindBiasedFixedOrderAdvectionScheme{N, FT, S <: AbstractUpwindBiasedAdvectionScheme} <: AbstractUpwindBiasedAdvectionScheme{N, FT}
    scheme::S
    function UpwindBiasedFixedOrderAdvectionScheme{N, FT}(scheme::S) where {N, FT, S<:AbstractUpwindBiasedAdvectionScheme}
        return new{N, FT, S}(scheme)
    end
end

fixed_order_scheme(scheme::Centered{N, FT}) where {N, FT} = CenteredFixedOrderAdvectionScheme{N, FT}(scheme)

fixed_order_scheme(scheme::UpwindBiased{N, FT}) where {N, FT} = UpwindBiasedFixedOrderAdvectionScheme{N, FT}(scheme)

fixed_order_scheme(scheme::WENO{N, FT}) where {N, FT} = UpwindBiasedFixedOrderAdvectionScheme{N, FT}(scheme)

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

const FixedOrderAdvectionScheme = Union{CenteredFixedOrderAdvectionScheme, UpwindBiasedFixedOrderAdvectionScheme}
# Overload all interpolation functions to skip any boundary checks
for bias in (:symmetric, :biased)
    for (d, (ξ, FG)) in enumerate(zip((:x, :y, :z), (:XFG, :YFG, :ZFG)))

        code = [:ᵃ, :ᵃ, :ᵃ]

        for loc in (:ᶜ, :ᶠ)
            code[d] = loc
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            _interp = Symbol(:_, interp)

            @eval begin
                @inline function $_interp(i, j, k, grid::AbstractGrid, scheme::FixedOrderAdvectionScheme, args...)
                    return $interp(i, j, k, grid, scheme.scheme, args...)
                end

                @inline $_interp(i, j, k, grid::$FG, ::FixedOrderAdvectionScheme, ψ, args...) = @inbounds ψ[i, j, k]
                @inline $_interp(i, j, k, grid::$FG, ::FixedOrderAdvectionScheme, ψ::Callable, args...) = @inbounds ψ(i, j, k, grid, args...)
                @inline $_interp(i, j, k, grid::$FG, ::FixedOrderAdvectionScheme, ψ::Callable, ::AS, args...) = @inbounds ψ(i, j, k, grid, args...)
            end
        end
    end
end
