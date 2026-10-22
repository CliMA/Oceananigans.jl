using Oceananigans.ImmersedBoundaries: ImmersedTopIBG

#####
##### Tracer advective fluxes vanish through an immersed top and bottom
#####

@inline _advective_tracer_flux_x(i, j, k, ibg::ImmersedTopIBG, scheme, U, c) =
    conditional_flux_fcc(i, j, k, ibg, zero(ibg), advective_tracer_flux_x(i, j, k, ibg, scheme, TimeSteppers.time_discretization(scheme), U, c))

@inline _advective_tracer_flux_y(i, j, k, ibg::ImmersedTopIBG, scheme, V, c) =
    conditional_flux_cfc(i, j, k, ibg, zero(ibg), advective_tracer_flux_y(i, j, k, ibg, scheme, TimeSteppers.time_discretization(scheme), V, c))

@inline _advective_tracer_flux_z(i, j, k, ibg::ImmersedTopIBG, scheme, W, c) =
    conditional_flux_ccf(i, j, k, ibg, zero(ibg), advective_tracer_flux_z(i, j, k, ibg, scheme, TimeSteppers.time_discretization(scheme), W, c))

@inline _advective_tracer_flux_x(i, j, k, ibg::ImmersedTopIBG, ::Nothing, U, c) = zero(ibg)
@inline _advective_tracer_flux_y(i, j, k, ibg::ImmersedTopIBG, ::Nothing, V, c) = zero(ibg)
@inline _advective_tracer_flux_z(i, j, k, ibg::ImmersedTopIBG, ::Nothing, W, c) = zero(ibg)

@inline _advective_tracer_flux_x(i, j, k, ibg::ImmersedTopIBG, scheme::FluxFormAdvection, U, c) = _advective_tracer_flux_x(i, j, k, ibg, scheme.x, U, c)
@inline _advective_tracer_flux_y(i, j, k, ibg::ImmersedTopIBG, scheme::FluxFormAdvection, V, c) = _advective_tracer_flux_y(i, j, k, ibg, scheme.y, V, c)
@inline _advective_tracer_flux_z(i, j, k, ibg::ImmersedTopIBG, scheme::FluxFormAdvection, W, c) = _advective_tracer_flux_z(i, j, k, ibg, scheme.z, W, c)
