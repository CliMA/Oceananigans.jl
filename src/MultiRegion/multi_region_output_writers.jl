using Oceananigans.Fields: compute_at!

import Oceananigans.OutputWriters: fetch_output,
                                   construct_output,
                                   serializeproperty!

# This is working just fine at the moment?
# But it will be veeeeery slow, as reconstruct_global_field is not 
# a performant operation
function fetch_output(mrf::MultiRegionField, model)
    field = reconstruct_global_field(mrf)
    compute_at!(field, model.clock.time)
    return parent(field)
end
  
function construct_output(mrf::MultiRegionField, grid, user_indices, with_halos)
    # TODO: support non-default indices I guess
    # for that we have to figure out how to partition indices, eg user_indices is "global"
    # indices = output_indices(user_output, grid, user_indices, with_halos)

    indices = (:, :, user_indices[3]) # sorry user

    return construct_output(mrf, indices)
end

function serializeproperty!(file, location, mrf::MultiRegionField{LX, LY, LZ}) where {LX, LY, LZ}
    p = reconstruct_global_field(mrf)
    serializeproperty!(file, location * "/location", (LX(), LY(), LZ()))
    serializeproperty!(file, location * "/data", parent(p))
    serializeproperty!(file, location * "/boundary_conditions", p.boundary_conditions)
  
    return nothing
end

function serializeproperty!(file, location, mrg::MultiRegionGrid) 
    file[location] = on_architecture(CPU(), reconstruct_global_grid(mrg))
end
