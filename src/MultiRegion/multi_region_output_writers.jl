import Oceananigans.OutputWriters: 
                        fetch_output,
                        construct_output,
                        serializeproperty!

function fetch_output(mrf::MultiRegionField, model)
    field = reconstruct_global_field(mrf)
    compute_at!(field, time(model))
    return parent(field)
  end
  
  function construct_output(mrf::MultiRegionField, grid, user_indices, with_halos)
    user_output = reconstruct_global_field(mrf)
    grid = user_output.grid
    indices = output_indices(user_output, grid, user_indices, with_halos)
    return construct_output(user_output, indices)
  end
  
  function serializeproperty!(file, location, mrf::MultiRegionField{LX, LY, LZ}) where {LX, LY, LZ}
    p = reconstruct_global_field(mrf)
    serializeproperty!(file, location * "/location", (LX(), LY(), LZ()))
    serializeproperty!(file, location * "/data", parent(p))
    serializeproperty!(file, location * "/boundary_conditions", p.boundary_conditions)
  end
  