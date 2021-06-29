using Oceananigans.TurbulenceClosures: IsotropicDiffusivity
using Oceananigans.BuoyancyModels

"""
    NonDimensionalIncompressibleModel(; grid, Re, Pr=0.7, Ro=Inf, kwargs...)

Construct a "Non-dimensional" `Model` on `grid` with the four non-dimensional numbers:


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
function NonDimensionalIncompressibleModel(; grid, Re, Pr=0.7, Ro=Inf,
                                           buoyancy = BuoyancyTracer(),
                                           coriolis = FPlane(eltype(grid), f=1/Ro),
                                           closure = IsotropicDiffusivity(eltype(grid), ν=1/Re, κ=1/(Pr*Re)),
                                           kwargs...)

    return IncompressibleModel(grid=grid,
                               closure=closure,
                               coriolis=coriolis,
                               tracers=:b,
                               buoyancy=buoyancy,
                               kwargs...)
end
