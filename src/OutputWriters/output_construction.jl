using Oceananigans.Fields: validate_index, Reduction
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Grids: default_indices

restrict_to_interior(::Colon, loc, topo, N) = interior_indices(loc, topo, N)

function restrict_to_interior(index::UnitRange, loc, topo, N)
    from = max(first(index), 1)
    to = min(last(index), last(interior_indices(loc, topo, N)))
    return UnitRange(from, to)
end

#####
##### Function output fallback
#####

function construct_output(output, grid, indices, with_halos)
    if !(indices isa typeof(default_indices(3)))
        output_type = output isa Function ? "Function" : ""
        @warn "Cannot slice $output_type $output with $indices: output will be unsliced."
    end

    return output
end

#####
##### Support for Field, Reduction, and AbstractOperation outputs
#####

function output_indices(output::AbstractField, grid, indices, with_halos)
    indices = validate_indices(indices, location(output), grid)

    if !with_halos
        loc = location(output)
        topo = topology(grid)
        maybe_chopped_indices = restrict_to_interior.(indices, loc, topo, size(grid))
    end

    return maybe_chopped_indices
end


function construct_output(user_output::AbstractField, grid, user_indices, with_halos)
    indices = output_indices(out, grid, user_indices, with_halos)
    return construct_output(user_output, indices)
end

construct_output(user_output::Field, indices) = view(user_output, indices...)
construct_output(user_output::Reduction, indices) = Field(user_output; indices)
construct_output(user_output::AbstractOperation, indices) = Field(user_output; indices)

#####
##### Time-averaging
#####

function construct_output(averaged_output::WindowedTimeAverage{<:Field}, grid, indices, with_halos)
    output = construct_output(averaged_output.operand, grid, indices, with_halos)
    return WindowedTimeAverage(output; schedule=averaged_output.schedule)
end

