using Oceananigans.ImmersedBoundaries: immersed_peripheral_node
using Oceananigans.Fields: ZeroField

const IBG = ImmersedBoundaryGrid

const c = Center()
const f = Face()

"""
$(TYPEDSIGNATURES)

Return either

    i) The boundary flux `qᴮ` if the node condition `nc` is true (default: `nc = immersed_peripheral_node`), or
    ii) The interior flux `qᴵ` otherwise.

This can be used either to condition intrinsic flux functions, or immersed boundary flux functions.
"""
@inline function conditional_flux(i, j, k, ibg, ℓx, ℓy, ℓz, q_boundary, q_interior)
    on_immersed_periphery = immersed_peripheral_node(i, j, k, ibg, ℓx, ℓy, ℓz)
    return ifelse(on_immersed_periphery, q_boundary, q_interior)
end

# Conveniences
@inline conditional_flux_ccc(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, c, c, c, qᴮ, qᴵ)
@inline conditional_flux_ffc(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, f, f, c, qᴮ, qᴵ)
@inline conditional_flux_fcf(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, f, c, f, qᴮ, qᴵ)
@inline conditional_flux_cff(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, c, f, f, qᴮ, qᴵ)

@inline conditional_flux_fcc(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, f, c, c, qᴮ, qᴵ)
@inline conditional_flux_cfc(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, c, f, c, qᴮ, qᴵ)
@inline conditional_flux_ccf(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, c, c, f, qᴮ, qᴵ)

#####
##### Immersed Advective fluxes
#####

# Fallback for `nothing` advection
@inline _advective_tracer_flux_x(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
@inline _advective_tracer_flux_y(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
@inline _advective_tracer_flux_z(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)

# Fallback for `nothing` advection
@inline _advective_momentum_flux_Uu(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
@inline _advective_momentum_flux_Uv(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
@inline _advective_momentum_flux_Uw(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
@inline _advective_momentum_flux_Vu(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
@inline _advective_momentum_flux_Vv(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
@inline _advective_momentum_flux_Vw(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
@inline _advective_momentum_flux_Wu(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
@inline _advective_momentum_flux_Wv(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
@inline _advective_momentum_flux_Ww(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)

# dx(uu), dy(vu), dz(wu)
# ccc,    ffc,    fcf
@inline _advective_momentum_flux_Uu(i, j, k, ibg::IBG, scheme, U, u) = conditional_flux_ccc(i, j, k, ibg, zero(ibg), advective_momentum_flux_Uu(i, j, k, ibg, scheme, TimeSteppers.time_discretization(scheme), U, u))
@inline _advective_momentum_flux_Vu(i, j, k, ibg::IBG, scheme, V, u) = conditional_flux_ffc(i, j, k, ibg, zero(ibg), advective_momentum_flux_Vu(i, j, k, ibg, scheme, TimeSteppers.time_discretization(scheme), V, u))
@inline _advective_momentum_flux_Wu(i, j, k, ibg::IBG, scheme, W, u) = conditional_flux_fcf(i, j, k, ibg, zero(ibg), advective_momentum_flux_Wu(i, j, k, ibg, scheme, TimeSteppers.time_discretization(scheme), W, u))

# dx(uv), dy(vv), dz(wv)
# ffc,    ccc,    cff
@inline _advective_momentum_flux_Uv(i, j, k, ibg::IBG, scheme, U, v) = conditional_flux_ffc(i, j, k, ibg, zero(ibg), advective_momentum_flux_Uv(i, j, k, ibg, scheme, TimeSteppers.time_discretization(scheme), U, v))
@inline _advective_momentum_flux_Vv(i, j, k, ibg::IBG, scheme, V, v) = conditional_flux_ccc(i, j, k, ibg, zero(ibg), advective_momentum_flux_Vv(i, j, k, ibg, scheme, TimeSteppers.time_discretization(scheme), V, v))
@inline _advective_momentum_flux_Wv(i, j, k, ibg::IBG, scheme, W, v) = conditional_flux_cff(i, j, k, ibg, zero(ibg), advective_momentum_flux_Wv(i, j, k, ibg, scheme, TimeSteppers.time_discretization(scheme), W, v))

# dx(uw), dy(vw), dz(ww)
# fcf,    cff,    ccc
@inline _advective_momentum_flux_Uw(i, j, k, ibg::IBG, scheme, U, w) = conditional_flux_fcf(i, j, k, ibg, zero(ibg), advective_momentum_flux_Uw(i, j, k, ibg, scheme, TimeSteppers.time_discretization(scheme), U, w))
@inline _advective_momentum_flux_Vw(i, j, k, ibg::IBG, scheme, V, w) = conditional_flux_cff(i, j, k, ibg, zero(ibg), advective_momentum_flux_Vw(i, j, k, ibg, scheme, TimeSteppers.time_discretization(scheme), V, w))
@inline _advective_momentum_flux_Ww(i, j, k, ibg::IBG, scheme, W, w) = conditional_flux_ccc(i, j, k, ibg, zero(ibg), advective_momentum_flux_Ww(i, j, k, ibg, scheme, TimeSteppers.time_discretization(scheme), W, w))

# Disambiguation for `FluxForm` momentum fluxes....
@inline _advective_momentum_flux_Uu(i, j, k, ibg::IBG, scheme::FluxFormAdvection, U, u) = _advective_momentum_flux_Uu(i, j, k, ibg, scheme.x, U, u)
@inline _advective_momentum_flux_Vu(i, j, k, ibg::IBG, scheme::FluxFormAdvection, V, u) = _advective_momentum_flux_Vu(i, j, k, ibg, scheme.y, V, u)
@inline _advective_momentum_flux_Wu(i, j, k, ibg::IBG, scheme::FluxFormAdvection, W, u) = _advective_momentum_flux_Wu(i, j, k, ibg, scheme.z, W, u)

@inline _advective_momentum_flux_Uv(i, j, k, ibg::IBG, scheme::FluxFormAdvection, U, v) = _advective_momentum_flux_Uv(i, j, k, ibg, scheme.x, U, v)
@inline _advective_momentum_flux_Vv(i, j, k, ibg::IBG, scheme::FluxFormAdvection, V, v) = _advective_momentum_flux_Vv(i, j, k, ibg, scheme.y, V, v)
@inline _advective_momentum_flux_Wv(i, j, k, ibg::IBG, scheme::FluxFormAdvection, W, v) = _advective_momentum_flux_Wv(i, j, k, ibg, scheme.z, W, v)

@inline _advective_momentum_flux_Uw(i, j, k, ibg::IBG, scheme::FluxFormAdvection, U, w) = _advective_momentum_flux_Uw(i, j, k, ibg, scheme.x, U, w)
@inline _advective_momentum_flux_Vw(i, j, k, ibg::IBG, scheme::FluxFormAdvection, V, w) = _advective_momentum_flux_Vw(i, j, k, ibg, scheme.y, V, w)
@inline _advective_momentum_flux_Ww(i, j, k, ibg::IBG, scheme::FluxFormAdvection, W, w) = _advective_momentum_flux_Ww(i, j, k, ibg, scheme.z, W, w)

# Disambiguation for `FluxForm` tracer fluxes....
@inline _advective_tracer_flux_x(i, j, k, ibg::IBG, scheme::FluxFormAdvection, U, c) =
        _advective_tracer_flux_x(i, j, k, ibg, scheme.x, U, c)

@inline _advective_tracer_flux_y(i, j, k, ibg::IBG, scheme::FluxFormAdvection, V, c) =
        _advective_tracer_flux_y(i, j, k, ibg, scheme.y, V, c)

@inline _advective_tracer_flux_z(i, j, k, ibg::IBG, scheme::FluxFormAdvection, W, c) =
        _advective_tracer_flux_z(i, j, k, ibg, scheme.z, W, c)
