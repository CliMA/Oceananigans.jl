using Oceananigans.Fields: validate_index
using Oceananigans.Grids: default_indices


interior_restrict_index(::Colon, loc, topo, N) = interior_indices(loc, topo, N)

function interior_restrict_index(index::UnitRange, loc, topo, N)
    left = max(index[1], 1)
    right = min(index[end], interior_indices(loc, topo, N)[end])
    return UnitRange(left, right)
end

function construct_output(output, grid, indices, with_halos)
    if !(indices isa default_indices(ndims(output)))
        output_type = output isa Function ? "Function" : ""
        @warn "Cannot slice $output_type $output with $indices: output will be unsliced."
    end

    return output
end

function construct_output(output::Field, grid, indices, with_halos)
    indices = validate_index.(indices, location(output), size(output))

    if with_halos
        return view(output, indices...)
    else
        loc = location(output)
        topo = topology(grid)
        interior_indices = interior_restrict_index.(indices, loc, topo, size(grid))
        return view(output, interior_indices...)
    end
end

function construct_output(averaged_output::WindowedTimeAverage{<:Field}, grid, indices, with_halos)
    output = construct_output(averaged_output.operand, grid, indices, with_halos)
    return WindowedTimeAverage(output; schedule=averaged_output.schedule)
end

