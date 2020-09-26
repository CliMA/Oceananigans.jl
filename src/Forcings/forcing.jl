"""
    Forcing(func; parameters=nothing, field_dependencies=(), discrete_form=false)

Returns a forcing function.
"""
function Forcing(func; parameters=nothing, field_dependencies=(), discrete_form=false)
    if discrete_form
        return DiscreteForcing(func; parameters=parameters)
    else
        return ContinuousForcing(func; parameters=parameters, field_dependencies=field_dependencies)
    end
end
