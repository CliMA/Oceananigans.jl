using Oceananigans.TimeSteppers: TimeSteppers,
                                 AbstractTimeDiscretization,
                                 ExplicitTimeDiscretization,
                                 VerticallyImplicitTimeDiscretization,
                                 AdaptiveVerticallyImplicitDiscretization

const AdaptiveImplicitVerticalAdvection = AbstractAdvectionScheme{<:Any, <:Any, <:AdaptiveVerticallyImplicitDiscretization}

const AIVA = AdaptiveImplicitVerticalAdvection
const ATD = AbstractTimeDiscretization

@inline TimeSteppers.time_discretization(scheme::AbstractAdvectionScheme) = scheme.time_discretization
