using Oceananigans.Grids: RectilinearGrid, LatitudeLongitudeGrid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.BuoyancyFormulations: BuoyancyForce, BuoyancyTracer, SeawaterBuoyancy, LinearEquationOfState

using SeawaterPolynomials: BoussinesqEquationOfState

import Oceananigans.OutputWriters: default_output_attributes

const BoussinesqSeawaterBuoyancy = SeawaterBuoyancy{FT, <:BoussinesqEquationOfState, T, S} where {FT, T, S}
const BuoyancyBoussinesqEOSModel = BuoyancyForce{<:BoussinesqSeawaterBuoyancy, g} where {g}

#####
##### Specialized velocity attributes
#####

default_velocity_attributes(::RectilinearGrid) = Dict(
    "u" => Dict("long_name" => "Velocity in the +x-direction.", "units" => "m/s"),
    "v" => Dict("long_name" => "Velocity in the +y-direction.", "units" => "m/s"),
    "w" => Dict("long_name" => "Velocity in the +z-direction.", "units" => "m/s"))

default_velocity_attributes(::LatitudeLongitudeGrid) = Dict(
    "u"            => Dict("long_name" => "Velocity in the zonal direction (+ = east).", "units" => "m/s"),
    "v"            => Dict("long_name" => "Velocity in the meridional direction (+ = north).", "units" => "m/s"),
    "w"            => Dict("long_name" => "Velocity in the vertical direction (+ = up).", "units" => "m/s"),
    "displacement" => Dict("long_name" => "Sea surface height displacement", "units" => "m"))

default_velocity_attributes(ibg::ImmersedBoundaryGrid) = default_velocity_attributes(ibg.underlying_grid)

#####
##### Specialized tracer attributes
#####

default_tracer_attributes(::BuoyancyForce{<:BuoyancyTracer}) = Dict("b" => Dict("long_name" => "Buoyancy", "units" => "m/s²"))

default_tracer_attributes(::BuoyancyForce{<:SeawaterBuoyancy{FT, <:LinearEquationOfState}}) where FT = Dict(
    "T" => Dict("long_name" => "Temperature", "units" => "°C"),
    "S" => Dict("long_name" => "Salinity",    "units" => "practical salinity unit (psu)"))

default_tracer_attributes(::BuoyancyBoussinesqEOSModel) = Dict(
    "T" => Dict("long_name" => "Conservative temperature", "units" => "°C"),
    "S" => Dict("long_name" => "Absolute salinity",        "units" => "g/kg"))

#####
##### Specialized default_output_attributes
#####

function default_output_attributes(model::Union{NonhydrostaticModel, HydrostaticFreeSurfaceModel})
    velocity_attrs = default_velocity_attributes(model.grid)
    tracer_attrs = default_tracer_attributes(model.buoyancy)
    return merge(velocity_attrs, tracer_attrs)
end

function default_output_attributes(model::ShallowWaterModel)
    return default_velocity_attributes(model.grid)
end
