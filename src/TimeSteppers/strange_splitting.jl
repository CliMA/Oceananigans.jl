using Oceananigans.Fields: FunctionField, location
using Oceananigans.Utils: @apply_regionally, apply_regionally!

"""
    StrangeSplittingTimeStepper

Strange splitting 
"""
struct StrangeSplittingTimeStepper{PT, BT} <: AbstractTimeStepper
            physics :: PT
    biogeochemistry :: BT

    function StrangeSplittingTimeStepper(grid, tracers;
                                         implicit_solver = nothing,
                                         physics::PT = RungeKutta3TimeStepper(grid, tracers; implicit_solver),
                                         biogeochemistry::BT = RungeKutta3TimeStepper(grid, tracers)) where {PT, BT}
        isa(biogeochemistry, RungeKutta3TimeStepper) || @warn "RK3 is the only biogeochemistry timestepper currently supported"

        return new{PT, BT}(physics, biogeochemistry)
    end
end

reset!(ts::StrangeSplittingTimeStepper) = (reset!(ts.physics); reset!(ts.biogeochemistry))

"""
    time_step!(model::AbstractModel{<:StrangeSplittingTimeStepper}, Δt;)

"""
function time_step!(model::AbstractModel{<:StrangeSplittingTimeStepper}, Δt; callbacks=[])
    timesteppers = model.timestepper

    time_step_biogeochemistry!(timesteppers.biogeochemistry, model, Δt/2)

    time_step!(timesteppers.physics, model, Δt; callbacks)

    time_step_biogeochemistry!(timesteppers.biogeochemistry, model, Δt/2)
    
    return nothing
end

compute_bgc_with_physics(::StrangeSplittingTimeStepper) = false

timestepper_tendencies(timestepper::StrangeSplittingTimeStepper) = timestepper.physics.Gⁿ
timestepper_previous_tendencies(timestepper::StrangeSplittingTimeStepper) = timestepper.physics.G⁻

# TODO: implement step bgc tracers... - does this need to include surface fluxs? Probably, is that handelled by fill halo still?