using Oceananigans.ImmersedBoundaries: BottomAndTopIBG

#####
##### Tracer advective fluxes vanish through the immersed top and bottom
#####

@inline _advective_tracer_flux_x(i, j, k, ibg::BottomAndTopIBG, scheme, U, c) =
    conditional_flux_fcc(i, j, k, ibg, zero(ibg), advective_tracer_flux_x(i, j, k, ibg, scheme, TimeSteppers.time_discretization(scheme), U, c))

@inline _advective_tracer_flux_y(i, j, k, ibg::BottomAndTopIBG, scheme, V, c) =
    conditional_flux_cfc(i, j, k, ibg, zero(ibg), advective_tracer_flux_y(i, j, k, ibg, scheme, TimeSteppers.time_discretization(scheme), V, c))

@inline _advective_tracer_flux_z(i, j, k, ibg::BottomAndTopIBG, scheme, W, c) =
    conditional_flux_ccf(i, j, k, ibg, zero(ibg), advective_tracer_flux_z(i, j, k, ibg, scheme, TimeSteppers.time_discretization(scheme), W, c))

@inline _advective_tracer_flux_x(i, j, k, ibg::BottomAndTopIBG, ::Nothing, U, c) = zero(ibg)
@inline _advective_tracer_flux_y(i, j, k, ibg::BottomAndTopIBG, ::Nothing, V, c) = zero(ibg)
@inline _advective_tracer_flux_z(i, j, k, ibg::BottomAndTopIBG, ::Nothing, W, c) = zero(ibg)

@inline _advective_tracer_flux_x(i, j, k, ibg::BottomAndTopIBG, scheme::FluxFormAdvection, U, c) = _advective_tracer_flux_x(i, j, k, ibg, scheme.x, U, c)
@inline _advective_tracer_flux_y(i, j, k, ibg::BottomAndTopIBG, scheme::FluxFormAdvection, V, c) = _advective_tracer_flux_y(i, j, k, ibg, scheme.y, V, c)
@inline _advective_tracer_flux_z(i, j, k, ibg::BottomAndTopIBG, scheme::FluxFormAdvection, W, c) = _advective_tracer_flux_z(i, j, k, ibg, scheme.z, W, c)
