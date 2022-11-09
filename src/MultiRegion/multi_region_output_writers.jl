using Oceananigans.Fields: compute_at!, AbstractField

import Oceananigans.OutputWriters: 
                        fetch_output,
                        construct_output,
                        saveproperty!,
                        serializeproperties!,
                        reconstruct_field


using Oceananigans.OutputWriters: _saveproperty!

const MultiRegionAbstractField = AbstractField{<:Any, <:Any, <:Any, <:MultiRegionGrid}

# This is working just fine at the moment?
# But it will be veeeeery slow, as reconstruct_global_field is not 
# a performant operation
function fetch_output(mrf::MultiRegionField, model)
    compute!(mrf)
    field = reconstruct_global_field(mrf)
    return parent(field)
end
  
function construct_output(mrf::MultiRegionAbstractField, grid, user_indices, with_halos)
    # TODO: support non-default indices I guess
    # for that we have to figure out how to partition indices, eg user_indices is "global"
    # indices = output_indices(user_output, grid, user_indices, with_halos)
    indices = (:, :, user_indices[3]) # sorry user
  
    return Field(mrf; indices)
end

function serializeproperty!(file, location, mrf::MultiRegionAbstractField{LX, LY, LZ}) where {LX, LY, LZ}
    p = reconstruct_global_field(mrf)
    serializeproperty!(file, location * "/location", (LX(), LY(), LZ()))
    serializeproperty!(file, location * "/data", parent(p))
    serializeproperty!(file, location * "/boundary_conditions", p.boundary_conditions)
  
    return nothing
end

function serializeproperty!(file, location, mrg::MultiRegionGrid) 
    @show "I am here"
    file[location] = on_architecture(CPU(), reconstruct_global_grid(mrg))
end

reconstruct_field(mrf::MultiRegionAbstractField)   = reconstruct_global_field(mrf)
saveproperty!(file, address, mrg::MultiRegionGrid) = _saveproperty!(file, address, on_architecture(CPU(), reconstruct_global_grid(mrg)))
