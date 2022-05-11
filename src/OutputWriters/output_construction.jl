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

function output_indices(output::Union{AbstractField, Reduction}, grid, indices, with_halos)
    indices = validate_indices(indices, location(output), grid)

    if !with_halos # Maybe chop those indices
        loc = location(output)
        topo = topology(grid)
        indices = restrict_to_interior.(indices, loc, topo, size(grid))
    end

    return indices
end

function construct_output(user_output::Union{AbstractField, Reduction}, grid, user_indices, with_halos)
    indices = output_indices(user_output, grid, user_indices, with_halos)
    return construct_output(user_output, indices)
end


const WindowedData = OffsetArray{<:Any, <:Any, <:SubArray}
const WindowedField = Field{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:WindowedData}
const ComputedField = Field{<:Any, <:Any, <:Any, <:AbstractOperation}

function construct_output(user_output::WindowedField, grid, user_indices, with_halos)
    if !with_halos
        new_indices = restrict_to_interior.(user_output.indices, location(user_output), topology(grid), size(grid))
        return view(user_output, new_indices...)
    end

    return user_output
end

function construct_output(user_output::ComputedField, grid, user_indices, with_halos)
    if indices(user_output) == (Colon(), Colon(), Colon())
        return construct_output(user_output.operand, grid, user_indices, with_halos)
    else
        @info "Only change halos"
        if !with_halos
            new_indices = [ indexes == Colon() ? restrict_to_interior(indexes, loc, topo, grid_size) : indexes
                           for (indexes, loc, topo, grid_size) in zip(indices(user_output), location(user_output), topology(grid), size(grid)) ]
            @show new_indices summary(user_output.data)
            return view(user_output, new_indices...)
        end
    end

    return user_output
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

