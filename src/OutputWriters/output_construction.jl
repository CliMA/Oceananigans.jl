using Oceananigans.Fields: validate_indices, Reduction, indices
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Grids: default_indices

restrict_to_interior(::Colon, loc, topo, N) = interior_indices(loc, topo, N)
restrict_to_interior(::Colon, ::Type{Nothing}, topo, N) = UnitRange(1, 1)
restrict_to_interior(index::UnitRange, ::Type{Nothing}, topo, N) = UnitRange(1, 1)

function restrict_to_interior(index::UnitRange, loc, topo, N)
    from = max(first(index), 1)
    to = min(last(index), last(interior_indices(loc, topo, N)))
    return UnitRange(from, to)
end

#####
##### Function output fallback
#####

function construct_output(output, grid, output_writer_indices, with_halos)
    if !(output_writer_indices == (:, :, :))
        output_type = output isa Function ? "Function" : ""
        @warn "Cannot slice $output_type $output with $output_writer_indices: output will be unsliced."
    end

    return output
end

#####
##### Support for Field, Reduction, and AbstractOperation outputs
#####

function output_indices(output::Union{AbstractField, Reduction}, grid, output_writer_indices, with_halos)
    output_writer_indices = validate_indices(output_writer_indices, location(output), grid)

    if !with_halos # Maybe chop those indices
        loc = location(output)
        topo = topology(grid)
        output_writer_indices = restrict_to_interior.(output_writer_indices, loc, topo, size(grid))
    end

    return output_writer_indices
end

function construct_output(user_output::Union{AbstractField, Reduction}, grid, output_writer_indices, with_halos)
    output_writer_indices = output_indices(user_output, grid, output_writer_indices, with_halos)
    return construct_output(user_output, output_writer_indices)
end

# The easy cases...
construct_output(user_output::Reduction, indices) = Field(user_output; indices)
construct_output(user_output::AbstractOperation, indices) = Field(user_output; indices)

function construct_output(user_output::Field, output_writer_indices)
    if indices(user_output) === (:, :, :) # this field has default indices, let's re-index it:
        return view(user_output, output_writer_indices...)
    else # this field has non-default indices
        output_writer_indices != (:, :, :) && @warn "Ignoring output writer indices for output with indices $(indices(user_output))"
        return user_output
    end
end
    
#####
##### Time-averaging
#####

function construct_output(averaged_output::WindowedTimeAverage{<:Field}, grid, output_writer_indices, with_halos)
    output = construct_output(averaged_output.operand, grid, output_writer_indices, with_halos)
    return WindowedTimeAverage(output; schedule=averaged_output.schedule)
end

