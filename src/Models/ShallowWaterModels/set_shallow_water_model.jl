import Oceananigans.Fields: set!

using Oceananigans.Fields: ZeroField
using Oceananigans.Models.HydrostaticFreeSurfaceModels: IntrinsicCoordinateGrid,
                                                       set_from_extrinsic_velocities!
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: initialize_closure_fields!

function set!(model::ShallowWaterModel; kwargs...)
    set_u = :u in keys(kwargs)
    set_v = :v in keys(kwargs)

    if hasproperty(model.solution, :u) &&
       hasproperty(model.solution, :v) &&
       (set_u || set_v)
        u = set_u ? kwargs[:u] : ZeroField()
        v = set_v ? kwargs[:v] : ZeroField()

        if model.grid isa IntrinsicCoordinateGrid
            set_from_extrinsic_velocities!(model.solution,
                                           model.grid,
                                           u,
                                           v;
                                           set_u,
                                           set_v)
        else
            set!(model.solution; u, v)
        end
    end

    for (fldname, value) in kwargs
        fldname in (:u, :v) && continue

        if fldname ∈ propertynames(model.solution)
            ϕ = getproperty(model.solution, fldname)
        elseif fldname ∈ propertynames(model.tracers)
            ϕ = getproperty(model.tracers, fldname)
        else
            throw(ArgumentError("name $fldname not found in model.solution or model.tracers."))
        end
        set!(ϕ, value)
    end

    update_state!(model)
    initialize_closure_fields!(model.closure_fields, model.closure, model)

    return nothing
end
