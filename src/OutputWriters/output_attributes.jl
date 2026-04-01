#####
##### Variable attributes
#####

using .BuoyancyFormulations: BuoyancyForce, BuoyancyTracer, SeawaterBuoyancy, LinearEquationOfState
using .Models: ShallowWaterModel

using SeawaterPolynomials: BoussinesqEquationOfState

const BoussinesqSeawaterBuoyancy = SeawaterBuoyancy{FT, <:BoussinesqEquationOfState, T, S} where {FT, T, S}
const BuoyancyBoussinesqEOSModel = BuoyancyForce{<:BoussinesqSeawaterBuoyancy, g} where {g}

OutputWriters.default_velocity_attributes(::RectilinearGrid) = Dict(
    "u" => Dict("long_name" => "Velocity in the +x-direction.", "units" => "m/s"),
    "v" => Dict("long_name" => "Velocity in the +y-direction.", "units" => "m/s"),
    "w" => Dict("long_name" => "Velocity in the +z-direction.", "units" => "m/s"))

OutputWriters.default_velocity_attributes(::LatitudeLongitudeGrid) = Dict(
    "u"            => Dict("long_name" => "Velocity in the zonal direction (+ = east).", "units" => "m/s"),
    "v"            => Dict("long_name" => "Velocity in the meridional direction (+ = north).", "units" => "m/s"),
    "w"            => Dict("long_name" => "Velocity in the vertical direction (+ = up).", "units" => "m/s"),
    "displacement" => Dict("long_name" => "Sea surface height displacement", "units" => "m"))

OutputWriters.default_velocity_attributes(ibg::ImmersedBoundaryGrid) = OutputWriters.default_velocity_attributes(ibg.underlying_grid)

OutputWriters.default_tracer_attributes(::Nothing) = Dict()

OutputWriters.default_tracer_attributes(::BuoyancyForce{<:BuoyancyTracer}) = Dict("b" => Dict("long_name" => "Buoyancy", "units" => "m/s²"))

OutputWriters.default_tracer_attributes(::BuoyancyForce{<:SeawaterBuoyancy{FT, <:LinearEquationOfState}}) where FT = Dict(
    "T" => Dict("long_name" => "Temperature", "units" => "°C"),
    "S" => Dict("long_name" => "Salinity",    "units" => "practical salinity unit (psu)"))

OutputWriters.default_tracer_attributes(::BuoyancyBoussinesqEOSModel) = Dict("T" => Dict("long_name" => "Conservative temperature", "units" => "°C"),
                                                                             "S" => Dict("long_name" => "Absolute salinity",        "units" => "g/kg"))

function OutputWriters.default_output_attributes(model)
    velocity_attrs = OutputWriters.default_velocity_attributes(model.grid)
    buoyancy = model isa ShallowWaterModel ? nothing : model.buoyancy
    tracer_attrs = OutputWriters.default_tracer_attributes(buoyancy)
    return merge(velocity_attrs, tracer_attrs)
end

#####
##### Saving schedule metadata as global attributes
#####

OutputWriters.add_schedule_metadata!(attributes, schedule) = nothing

function OutputWriters.add_schedule_metadata!(global_attributes, schedule::IterationInterval)
    global_attributes["schedule"] = "IterationInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output iteration interval"] = "Output was saved every $(schedule.interval) iteration(s)."

    return nothing
end

function OutputWriters.add_schedule_metadata!(global_attributes, schedule::TimeInterval)
    global_attributes["schedule"] = "TimeInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output time interval"] = "Output was saved every $(prettytime(schedule.interval))."

    return nothing
end

function OutputWriters.add_schedule_metadata!(global_attributes, schedule::WallTimeInterval)
    global_attributes["schedule"] = "WallTimeInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output time interval"] =
        "Output was saved every $(prettytime(schedule.interval))."

    return nothing
end

function OutputWriters.add_schedule_metadata!(global_attributes, schedule::AveragedTimeInterval)
    global_attributes["schedule"] = "AveragedTimeInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output time interval"] = "Output was time-averaged and saved every $(prettytime(schedule.interval))."

    global_attributes["time_averaging_window"] = schedule.window
    global_attributes["time averaging window"] = "Output was time averaged with a window size of $(prettytime(schedule.window))"

    global_attributes["time_averaging_stride"] = schedule.stride
    global_attributes["time averaging stride"] = "Output was time averaged with a stride of $(schedule.stride) iteration(s) within the time averaging window."

    return nothing
end
