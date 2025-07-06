using Oceananigans.Architectures: architecture
using Oceananigans.Fields: compute_at!
using Oceananigans.OutputWriters: _saveproperty!

import Oceananigans.OutputWriters: fetch_output,
                                   convert_output,
                                   construct_output,
                                   saveproperty!,
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

saveproperty!(file, address, p::Union{MultiRegionObject, MultiRegionField}) = _saveproperty!(file, address, p)

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

convert_output(mo::MultiRegionObject, writer)
    MultiRegionObject(mo.backend, Tuple(convert(writer.array_type, obj) for obj in mo.regional_objects))

function construct_output(csf::CubedSphereField{LX, LY, LZ}, grid::ConformalCubedSphereGridOfSomeKind, user_indices,
                          with_halos) where {LX, LY, LZ}
    multi_region_indices = output_indices(csf, grid, user_indices, with_halos)
    indices = getregion(multi_region_indices, 1)

    return Field(csf; indices, NamedTuple()...)
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
