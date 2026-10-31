#####
##### Interface for radiation absorbed below the surface
#####

"""
    penetrative_radiation(forcing)

Return the radiation absorbed below the surface that `forcing` applies, or `nothing`.
"""
penetrative_radiation(forcing) = nothing

function penetrative_radiation(forcings::Union{Tuple, NamedTuple})
    for forcing in forcings
        radiation = penetrative_radiation(forcing)
        isnothing(radiation) || return radiation
    end
    return nothing
end

"""
    surface_radiative_buoyancy_flux(i, j, grid, radiation, buoyancy, fields)

Return the buoyancy flux of `radiation` through the surface of column `(i, j)`, positive upward.
"""
@inline surface_radiative_buoyancy_flux(i, j, grid, ::Nothing, buoyancy, fields) = zero(grid)

"""
    transmitted_fraction(radiation, i, j, grid, d)

Return the fraction of the surface radiative flux of `radiation` that reaches the depth `d` in column `(i, j)`.
"""
function transmitted_fraction end

"""
    transmitted_fraction_derivative(radiation, i, j, grid, d)

Return the derivative of `transmitted_fraction` with respect to the depth `d` in column `(i, j)`.
"""
function transmitted_fraction_derivative end

"""
    transmitted_thickness(radiation, i, j, grid, h)

Return the integral of `transmitted_fraction` from the surface to the depth `h` in column `(i, j)`.
"""
function transmitted_thickness end
