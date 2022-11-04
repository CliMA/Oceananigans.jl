module Biogeochemistry

"""Ensure that `tracers' contains biogeochemical tracers."""
validate_biogeochemistry(tracers, ::Nothing) = nothing

"""Return the biogeochemical forcing function for `tracer_name`."""
@inline zerofunction(i, j, k, grid, args...) = zero(grid)
get_biogeochemical_forcing(biogeochemistry, tracer_name) = zerofunction
 
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
