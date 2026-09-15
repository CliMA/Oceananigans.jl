using Oceananigans.BoundaryConditions: TidalHarmonics, tidal_ramp, equilibrium_tide
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ

@inline x_tidal_forcing(i, j, k, grid, clock, fields, p) =
    p.gravitational_acceleration * ∂xᶠᶜᶜ(i, j, k, grid, equilibrium_tide, clock, p.harmonics) *
    tidal_ramp(p.harmonics, clock.time)

@inline y_tidal_forcing(i, j, k, grid, clock, fields, p) =
    p.gravitational_acceleration * ∂yᶜᶠᶜ(i, j, k, grid, equilibrium_tide, clock, p.harmonics) *
    tidal_ramp(p.harmonics, clock.time)

"""
$(TYPEDSIGNATURES)

Forcing `(u = Fᵘ, v = Fᵛ)` of the astronomical tide of `harmonics` on a spherical grid, which enters
the momentum equations as ``+g ∇η_{eq}`` for the equilibrium elevation ``η_{eq}``. The gradient is
taken over the same cells as the model's pressure gradient, so a resting ocean with
``η = η_{eq}`` stays at rest.

The forcing is barotropic: the astronomical tide pulls uniformly over the column.

```jldoctest
using Oceananigans
using Dates

harmonics = TidalHarmonics(DateTime(2019, 4, 1))
keys(tidal_forcing(harmonics))

# output
(:u, :v)
```
"""
function tidal_forcing(harmonics::TidalHarmonics;
                       gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration)

    parameters = (; harmonics, gravitational_acceleration)
    u = Forcing(x_tidal_forcing; discrete_form = true, parameters)
    v = Forcing(y_tidal_forcing; discrete_form = true, parameters)

    return (; u, v)
end
