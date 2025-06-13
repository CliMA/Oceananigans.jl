using Oceananigans.Fields: compute_at!

import Oceananigans.OutputWriters: fetch_output,
                                   convert_output,
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

function serializeproperty!(file, location, mrf::MultiRegionField{LX, LY, LZ}) where {LX, LY, LZ}
    p = reconstruct_global_field(mrf)

    serializeproperty!(file, location * "/location", (LX(), LY(), LZ()))
    serializeproperty!(file, location * "/data", parent(p))
    serializeproperty!(file, location * "/boundary_conditions", p.boundary_conditions)

    return nothing
end

function serializeproperty!(file, location, mrg::MultiRegionGrids)
    file[location] = on_architecture(CPU(), reconstruct_global_grid(mrg))
end

#####
##### For a cubed sphere, we dump the entire field as is.
#####

function fetch_output(csf::CubedSphereField, model)
    compute_at!(csf, model.clock.time)
    return parent(csf)
end

convert_output(mo::MultiRegionObject, writer) = 
    MultiRegionObject(Tuple(convert(writer.array_type, obj) for obj in mo.regional_objects))

function construct_output(user_output::Union{AbstractField, Reduction}, grid::ConformalCubedSphereGridOfSomeKind,
                          user_indices, with_halos)
    multi_region_indices = output_indices(user_output, grid, user_indices, with_halos)
    indices = getregion(multi_region_indices, 1)

    # Don't compute AbstractOperations or Reductions
    additional_kw = user_output isa Field ? NamedTuple() : (; compute=false)

    return Field(user_output; indices, additional_kw...)
end

function serializeproperty!(file, location, csf::CubedSphereField{LX, LY, LZ}) where {LX, LY, LZ}
    csf_CPU = on_architecture(CPU(), csf)

    serializeproperty!(file, location * "/location", (LX(), LY(), LZ()))
    serializeproperty!(file, location * "/data", parent(csf_CPU))
    serializeproperty!(file, location * "/boundary_conditions", csf_CPU.boundary_conditions)

    return nothing
end

function serializeproperty!(file, location, csg::ConformalCubedSphereGridOfSomeKind)
    file[location] = on_architecture(CPU(), csg)
end
