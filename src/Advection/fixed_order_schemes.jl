fixed_order_scheme(scheme::Centered) = StaticCentered(scheme)

fixed_order_scheme(scheme::UpwindBiased) = StaticUpwindBiased(scheme)

fixed_order_scheme(scheme::WENO) = StaticWENO(scheme)

fixed_order_scheme(scheme::WENOVectorInvariant) = StaticWENOVectorInvariant(scheme)

#Fallback, maybe should use a warning if not implemented?
fixed_order_scheme(scheme) = scheme

"""Centered advection scheme with fixed order"""
const StaticCentered{N, FT} = Centered{N, FT, Nothing}

function StaticCentered(scheme::Centered{N, FT}) where {N, FT}
    return Centered(FT; order=scheme_order(scheme), buffer_scheme=nothing)
end

"""UpwindBiased advection scheme with fixed order"""
const StaticUpwindBiased{N, FT, SI} = UpwindBiased{N, FT, Nothing, SI}

function StaticUpwindBiased(scheme::UpwindBiased{N, FT}) where {N, FT}
    return UpwindBiased(FT; order=scheme_order(scheme), buffer_scheme=nothing)
end

"""WENO advection scheme with fixed order"""
const StaticWENO{N, FT, FT2, PP, SI} = WENO{N, FT, FT2, PP, Nothing, SI}

function StaticWENO(FT::DataType=Oceananigans.defaults.FloatType, FT2::DataType=Float32;
              order = 5,
              bounds = nothing)
    return WENO(FT, FT2; order=order, buffer_scheme=nothing, bounds=bounds)
end

function StaticWENO(weno::WENO{N, FT, FT2}) where {N, FT, FT2}
    return StaticWENO(FT, FT2; order=weno_order(weno))
end

StaticWENOVectorInvariant(scheme) = scheme

"""Construct a WENOVectorInvariant scheme with fixed order"""
function StaticWENOVectorInvariant(scheme::WENOVectorInvariant)

    return VectorInvariant(
        vorticity_scheme=StaticWENO(scheme.vorticity_scheme),
        vorticity_stencil=scheme.vorticity_stencil,
        vertical_advection_scheme=StaticWENO(scheme.vertical_advection_scheme),
        divergence_scheme=StaticWENO(scheme.divergence_scheme),
        kinetic_energy_gradient_scheme=scheme.kinetic_energy_gradient_scheme,
        upwinding=scheme.upwinding
    )
end
