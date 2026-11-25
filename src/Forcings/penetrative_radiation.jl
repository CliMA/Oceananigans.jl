#####
##### Interface for radiation absorbed below the surface
#####

"""
    AbstractRadiativeForcing

Forcing by a radiative flux that enters through the surface and is absorbed below it, heating the temperature, or the
buoyancy of a `BuoyancyTracer`, with the convergence of the flux. Subtypes extend `surface_radiative_flux`,
`transmitted_fraction`, `transmitted_fraction_derivative` and `transmitted_thickness`.
"""
abstract type AbstractRadiativeForcing end

"""
    penetrative_radiation(forcing)

Return the radiation absorbed below the surface that `forcing` applies, or `nothing`.
"""
penetrative_radiation(forcing) = nothing
penetrative_radiation(radiation::AbstractRadiativeForcing) = radiation

function penetrative_radiation(forcings::Union{Tuple, NamedTuple})
    for forcing in forcings
        radiation = penetrative_radiation(forcing)
        isnothing(radiation) || return radiation
    end
    return nothing
end

"""
    surface_radiative_flux(i, j, grid, radiation)

Return the flux of the forced tracer that `radiation` carries through the surface of column `(i, j)`, positive upward.
"""
function surface_radiative_flux end

"""
    transmitted_fraction(i, j, grid, radiation, d)

Return the fraction of the surface radiative flux of `radiation` that reaches the depth `d` in column `(i, j)`.
"""
function transmitted_fraction end

"""
    transmitted_fraction_derivative(i, j, grid, radiation, d)

Return the derivative of `transmitted_fraction` with respect to the depth `d` in column `(i, j)`.
"""
function transmitted_fraction_derivative end

"""
    transmitted_thickness(i, j, grid, radiation, h)

Return the integral of `transmitted_fraction` from the surface to the depth `h` in column `(i, j)`.
"""
function transmitted_thickness end

# Flux through the bottom face of cell k, positive upward; the bottom cell absorbs what reaches it
@inline function radiative_flux(i, j, k, grid, radiation)
    d = znode(i, j, size(grid, 3) + 1, grid, Center(), Center(), Face()) - znode(i, j, k, grid, Center(), Center(), Face())
    J = surface_radiative_flux(i, j, grid, radiation) * transmitted_fraction(i, j, grid, radiation, d)
    return ifelse(inactive_cell(i, j, k - 1, grid), zero(J), J)
end

@inline (radiation::AbstractRadiativeForcing)(i, j, k, grid, clock, fields) = - ∂zᶜᶜᶜ(i, j, k, grid, radiative_flux, radiation)

"""
    surface_radiative_buoyancy_flux(i, j, grid, radiation, buoyancy, fields)

Return the buoyancy flux of `radiation` through the surface of column `(i, j)`, positive upward.
"""
@inline surface_radiative_buoyancy_flux(i, j, grid, ::Nothing, buoyancy, fields) = zero(grid)
@inline surface_radiative_buoyancy_flux(i, j, grid, radiation::AbstractRadiativeForcing, ::BuoyancyTracerFormulation, fields) = surface_radiative_flux(i, j, grid, radiation)

@inline function surface_radiative_buoyancy_flux(i, j, grid, radiation::AbstractRadiativeForcing, buoyancy::BuoyancyForce{<:SeawaterBuoyancy}, fields)
    b = buoyancy.formulation
    T, S = get_temperature_and_salinity(b, fields)
    α = thermal_expansionᶜᶜᶠ(i, j, size(grid, 3) + 1, grid, b.equation_of_state, T, S)
    return b.gravitational_acceleration * α * surface_radiative_flux(i, j, grid, radiation)
end
