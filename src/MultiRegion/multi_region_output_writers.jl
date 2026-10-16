using Oceananigans.Models.HydrostaticFreeSurfaceModels: ZStarCoordinate
using Oceananigans.Fields: Fields, compute_at!, set!
using Oceananigans.OutputWriters: OutputWriters, WindowedTimeAverage, _saveproperty!, serializeproperty!,
                                  accumulate_average!, set_average!

# This is working just fine at the moment?
# But it will be veeeeery slow, as reconstruct_global_field is not
# a performant operation
function OutputWriters.fetch_output(mrf::MultiRegionField, model)
    field = reconstruct_global_field(mrf)
    compute_at!(field, model.clock.time)
    return parent(field)
end

Fields.set!(mrf::MultiRegionField, wta::WindowedTimeAverage) = set!(mrf, wta.result)

# Time averages of panel fields accumulate the parent arrays of each region
OutputWriters.set_average!(result::MultiRegionObject, value) = apply_regionally!(set_average!, result, value)

OutputWriters.accumulate_average!(result::MultiRegionObject, integrand::MultiRegionObject, previous_weight, integrand_weight) =
    apply_regionally!(accumulate_average!, result, integrand, previous_weight, integrand_weight)

OutputWriters.saveproperty!(file, address, p::Union{MultiRegionObject, MultiRegionField}) = _saveproperty!(file, address, p)

function OutputWriters.serializeproperty!(file, location, mrf::MultiRegionField{LX, LY, LZ}) where {LX, LY, LZ}
    p = reconstruct_global_field(mrf)

    serializeproperty!(file, location * "/location", (LX(), LY(), LZ()))
    serializeproperty!(file, location * "/data", parent(p))
    serializeproperty!(file, location * "/boundary_conditions", p.boundary_conditions)

    return nothing
end

function OutputWriters.serializeproperty!(file, location, mrg::MultiRegionGrids)
    file[location] = on_architecture(CPU(), reconstruct_global_grid(mrg))
end

#####
##### For a cubed sphere or a Yin-Yang grid, we dump the data of every panel as is.
#####

const PanelField{LX, LY, LZ} = Union{CubedSphereField{LX, LY, LZ}, YinYangField{LX, LY, LZ}}

function OutputWriters.fetch_output(field::PanelField, model)
    compute_at!(field, model.clock.time)
    return parent(field)
end

function OutputWriters.convert_output(mo::MultiRegionObject, writer)
    array_type = writer.array_type
    converted = Tuple(convert(array_type, obj) for obj in mo.regional_objects)
    return MultiRegionObject(converted)
end

function OutputWriters.construct_output(mrf::MultiRegionField{LX, LY, LZ}, user_indices, with_halos) where {LX, LY, LZ}
    multi_region_indices = output_indices(mrf, user_indices, with_halos)
    indices = getregion(multi_region_indices, 1)
    return Field(mrf; indices, NamedTuple()...)
end

function OutputWriters.serializeproperty!(file, location, field::PanelField{LX, LY, LZ}) where {LX, LY, LZ}
    cpu_field = on_architecture(CPU(), field)

    serializeproperty!(file, location * "/location", (LX(), LY(), LZ()))
    serializeproperty!(file, location * "/data", parent(cpu_field))
    serializeproperty!(file, location * "/boundary_conditions", cpu_field.boundary_conditions)

    return nothing
end

function OutputWriters.serializeproperty!(file, location, grid::Union{ConformalCubedSphereGridOfSomeKind, YinYangGridOfSomeKind})
    file[location] = on_architecture(CPU(), grid)
end

function Oceananigans.prognostic_state(coordinate::ZStarCoordinate, grid::MultiRegionGrids)
    states = Tuple(on_architecture(CPU(), Oceananigans.prognostic_state(coordinate, getregion(grid, r))) for r in regions(grid))
    return (; regions=states)
end

function Oceananigans.restore_prognostic_state!(coordinate::ZStarCoordinate, grid::MultiRegionGrids, from::NamedTuple{(:regions,)})
    for r in regions(grid)
        Oceananigans.restore_prognostic_state!(coordinate, getregion(grid, r), from.regions[r])
    end
    return coordinate
end

Oceananigans.restore_prognostic_state!(::ZStarCoordinate, ::MultiRegionGrids, from) =
    throw(ArgumentError("A z-star checkpoint of a multi-region grid must hold the vertical coordinate of every region."))
