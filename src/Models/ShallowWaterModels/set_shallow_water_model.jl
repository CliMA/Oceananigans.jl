import Oceananigans.Fields: set!

using Oceananigans.TimeSteppers: update_state!

function set!(model::ShallowWaterModel; kwargs...)
    for (fldname, value) in kwargs
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
    
    return nothing
end
