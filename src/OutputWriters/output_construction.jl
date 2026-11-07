using Oceananigans.Fields: validate_indices, Reduction, remapped_field
using Oceananigans.Grids: default_indices
using Oceananigans.Utils: @apply_regionally, getregion

restrict_to_interior(::Colon, loc, topo, N) = interior_indices(loc, topo, N)
restrict_to_interior(::Colon, ::Nothing, topo, N) = UnitRange(1, 1)
restrict_to_interior(index::Base.OneTo, loc, topo, N) = restrict_to_interior(UnitRange(index), loc, topo, N)
restrict_to_interior(::Base.OneTo, ::Nothing, topo, N) = UnitRange(1, 1)
restrict_to_interior(index::AbstractUnitRange, ::Nothing, topo, N) = UnitRange(1, 1)

function restrict_to_interior(index::AbstractUnitRange, loc, topo, N)
    from = max(first(index), 1)
    to = min(last(index), last(interior_indices(loc, topo, N)))
    return UnitRange(from, to)
end

#####
##### Function output fallback
#####

function construct_output(output, indices, with_halos)
    if !(indices isa typeof(default_indices(3)))
        output_type = output isa Function ? "Function" : ""
        @warn "Cannot slice $output_type $output with $indices: output will be unsliced."
    end

    return output
end

#####
##### Support for Field, Reduction, and AbstractOperation outputs
#####

intersect_indices(output, indices) = indices
intersect_indices(output::AbstractField, indices) = map(intersect_index_range, indices, Oceananigans.Fields.indices(output))

intersect_index_range(::Colon, ::Colon) = Colon()
intersect_index_range(range::UnitRange, ::Colon) = range
intersect_index_range(::Colon, range::UnitRange) = range
intersect_index_range(range1::UnitRange, range2::UnitRange) = intersect(range1, range2)

output_indices(output::AbstractField, indices, with_halos) = output_indices(output, output.grid, indices, with_halos)
output_indices(output::Reduction, indices, with_halos) = output_indices(output, output.operand.grid, indices, with_halos)

# Runs once per output at writer construction, so it is not specialized on the output type
Base.@nospecializeinfer function output_indices(@nospecialize(output::Union{AbstractField, Reduction}), @nospecialize(grid), indices, with_halos)
    indices = validate_indices(indices, location(output), grid)

    if !with_halos # Maybe chop those indices
        loc = map(instantiate, location(output))
        topo = map(instantiate, topology(grid))
        @apply_regionally indices = map(restrict_to_interior, indices, loc, topo, size(grid))
    end

    @apply_regionally intersected = intersect_indices(output, indices)

    return intersected
end

Base.@nospecializeinfer function construct_output(@nospecialize(user_output::Union{AbstractField, Reduction}), user_indices, with_halos)
    indices = output_indices(user_output, user_indices, with_halos)

    # Don't compute AbstractOperations or Reductions
    additional_kw = user_output isa Field ? NamedTuple() : (; compute=false)

    return Field(user_output; indices, additional_kw...)
end

construct_output(user_output::ConstantFields, user_indices, with_halos) = user_output

#####
##### Time-averaging
#####

function construct_output(averaged_output::WindowedTimeAverage{<:Field}, indices, with_halos)
    output = construct_output(averaged_output.operand, indices, with_halos)
    return WindowedTimeAverage(output; schedule=averaged_output.schedule)
end

#####
##### Time differentiation
#####

function construct_output(derivative::TimeDerivative, indices, with_halos)
    output = construct_output(derivative.operand, indices, with_halos)
    return TimeDerivative(output; expected_max_time_step_growth = derivative.expected_max_time_step_growth)
end

#####
##### Remapping onto the grid of the writer
#####

# Outputs are remapped onto `grid` with `remapped_field` unless they already live on it, or `grid` is `nothing`.
# `indices` and `with_halos` refer to `grid`, where the output is written.
same_grid(output_grid, grid) = output_grid === grid || output_grid == grid

function construct_output(output, indices, with_halos, grid)
    isnothing(grid) || @warn "Cannot remap $output onto $(summary(grid)): output will not be remapped."
    return construct_output(output, indices, with_halos)
end

Base.@nospecializeinfer function construct_output(@nospecialize(output::Union{AbstractField, Reduction}), user_indices, with_halos, grid)
    if isnothing(grid) || same_grid(Grids.grid(output), grid)
        return construct_output(output, user_indices, with_halos)
    end

    indices = output_indices(output, grid, user_indices, with_halos)
    return remapped_field(output, grid; indices = getregion(indices, 1))
end

construct_output(output::ConstantFields, indices, with_halos, grid) = output

construct_output(averaged_output::WindowedTimeAverage{<:Field}, indices, with_halos, grid) =
    WindowedTimeAverage(construct_output(averaged_output.operand, indices, with_halos, grid); schedule=averaged_output.schedule)

construct_output(derivative::TimeDerivative, indices, with_halos, grid) =
    TimeDerivative(construct_output(derivative.operand, indices, with_halos, grid);
                   expected_max_time_step_growth = derivative.expected_max_time_step_growth)
