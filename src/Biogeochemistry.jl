module Biogeochemistry

"""Ensure that `tracers' contains biogeochemical tracers."""
@inline function validate_biogeochemistry!(bgc, tracers)
    req_tracers = required_tracers(bgc)
    
    all(tracer ∈ tracers for tracer in req_tracers) ||
        error("$(req_tracers) must be among the list of tracers to use $(typeof(bgc).name.wrapper)")
    
    return nothing
end

required_tracers(::Nothing) = ()

"""
Update tracer tendencies.

Called at the end of calculate_tendencies!
"""
update_tendencies!(bgc, model) = nothing

"""
Update tracer tendencies.

Called at the end of calculate_tendencies!
"""
update_biogeochemical_state!(bgc, model) = nothing

"""Return the biogeochemical forcing for `val_tracer_name` when model is called."""
abstract type AbstractBiogeochemistry end
@inline (::AbstractBiogeochemistry)(i, j, k, grid, val_tracer_name, clock, fields) = zero(grid)

#=
# Example: simple NP model
function get_biogeochemical_forcing(::Nothing, tracer_name)
    if tracer_name === :N
        return nutrient_forcing
    elseif tracer_name === :P
        return plankton_forcing
    end
end
=#

#=
biogeochemical_forcing(i, j, k, grid, ::Val{id}, clock, biogeochemistry, fields)
=#

end # module
