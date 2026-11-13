using Oceananigans.ImmersedBoundaries: ImmersedTopIBG

#####
##### Vertical tracer advective fluxes vanish through an immersed top and bottom
#####

@inline _advective_tracer_flux_z(i, j, k, ibg::ImmersedTopIBG, scheme, W, c) =
    conditional_flux_ccf(i, j, k, ibg, zero(ibg), advective_tracer_flux_z(i, j, k, ibg, scheme, TimeSteppers.time_discretization(scheme), W, c))

@inline _advective_tracer_flux_z(i, j, k, ibg::ImmersedTopIBG, ::Nothing, W, c) = zero(ibg)

@inline _advective_tracer_flux_z(i, j, k, ibg::ImmersedTopIBG, scheme::FluxFormAdvection, W, c) = _advective_tracer_flux_z(i, j, k, ibg, scheme.z, W, c)
