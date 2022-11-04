module Biogeochemistry

"""Ensure that `tracers' contains biogeochemical tracers."""
validate_biogeochemistry(tracers, ::Nothing) = nothing

"""Return the biogeochemical forcing function for `tracer_name`."""
get_biogeochemical_forcing(::Nothing, tracer_name) = nothing
 
"""
Update tracer tendencies.

Called at the end of calculate_tendencies!
"""
update_tendencies!(::Nothing, model) = nothing

"""
Update tracer tendencies.

Called at the end of calculate_tendencies!
"""
update_biogeochemical_state!(::Nothing, model) = nothing

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
