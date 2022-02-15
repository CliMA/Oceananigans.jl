construct_output(output, grid, indices, with_halos) = output

interior_restrict_index(::Colon, loc, topo, N) = interior_indices(loc, topo, N)

function interior_restrict_index(index::UnitRange, loc, topo, N)
    left = max(index[1], 1)
    right = min(index[end], interior_indices(loc, topo, N)[end])
    return UnitRange(left, right)
end

function construct_output(output::Field, grid, indices, with_halos)
    if with_halos
        return view(output, indices...)
    else
        loc = location(output)
        topo = topology(grid)
        interior_indices = interior_restrict_index.(indices, loc, topo, size(grid))
        return view(output, interior_indices...)
    end
end

