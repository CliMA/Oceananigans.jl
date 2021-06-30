using Oceananigans.TurbulenceClosures: IsotropicDiffusivity
using Oceananigans.BuoyancyModels

"""
    NonDimensionalIncompressibleModel(; N, L, Re, Pr=0.7, Ro=Inf, float_type=Float64, kwargs...)

Construct a "Non-dimensional" `Model` with resolution `N`, domain extent `L`,
precision `float_type`, and the four non-dimensional numbers:

    * `Re = U λ / ν` (Reynolds number)
    * `Pr = U λ / κ` (Prandtl number)
    * `Ro = U / f λ` (Rossby number)

for characteristic velocity scale `U`, length-scale `λ`, viscosity `ν`,
tracer diffusivity `κ`, and Coriolis parameter `f`. BuoyancyModels is scaled
with `λ U²`, so that the Richardson number is `Ri=B`, where `B` is a
non-dimensional buoyancy scale set by the user via initial conditions or
forcing.

Note that `N`, `L`, and `Re` are required.

Additional `kwargs` are passed to the regular `IncompressibleModel` constructor.
"""
function NonDimensionalIncompressibleModel(; grid, float_type=Float64, Re, Pr=0.7, Ro=Inf,
    buoyancy = BuoyancyTracer(),
    coriolis = FPlane(float_type, f=1/Ro),
     closure = IsotropicDiffusivity(float_type, ν=1/Re, κ=1/(Pr*Re)),
    kwargs...)

    return IncompressibleModel(; float_type=float_type, grid=grid, closure=closure,
                               coriolis=coriolis, tracers=(:b,), buoyancy=buoyancy, kwargs...)
end
