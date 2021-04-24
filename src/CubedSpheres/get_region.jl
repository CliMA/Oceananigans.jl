using Oceananigans.Fields: location

using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface, PrescribedVelocityFields

get_region(obj, i) = obj # fallback

#####
##### The basics
#####

@inline get_region(grid::MultiRegionGrid, i) = @inbounds grid.regions[i]
@inline get_region(tup::MultiRegionTuple, i) = @inbounds tup[i]

#####
##### Fields
#####

@inline function get_region(field::AbstractMultiRegionField, i)
    LX, LY, LZ = location(field)

    # Should we define a new lower-level constructor for Field that doesn't call validate_field_data?
    return Field{LX, LY, LZ}(get_region(field.data, i),
                             field.architecture,
                             get_region(field.grid, i),
                             get_region(field.boundary_conditions, i))
end

@inline function get_region(reduced_field::MultiRegionAbstractReducedField, i)
    LX, LY, LZ = location(reduced_field)

    return ReducedField{LX, LY, LZ}(get_region(reduced_field.data, i),
                                    reduced_field.architecture,
                                    get_region(reduced_field.grid, i),
                                    reduced_field.dims,
                                    get_region(reduced_field.boundary_conditions, i))
end

#####
##### Recursion
#####

@inline get_region(t::Tuple, i) = Tuple(get_region(t_elem, i) for t_elem in t)
@inline get_region(nt::NamedTuple, i) = NamedTuple{keys(nt)}(get_region(nt_elem, i) for nt_elem in nt)

#####
##### Model-specific stuff
#####

@inline get_region(free_surface::ExplicitFreeSurface, i) =
    ExplicitFreeSurface(get_region(free_surface.η, i), free_surface.gravitational_acceleration)

@inline get_region(U::PrescribedVelocityFields, i) = PrescribedVelocityFields(get_region(U.u, i),
                                                                              get_region(U.v, i),
                                                                              get_region(U.w, i),
                                                                              U.parameters)

#####
##### For iterating!
#####

nregions(anything) = 1 # fallback
nregions(grid::MultiRegionGrid) = length(grid.regions)
nregions(field::AbstractMultiRegionField) = nregions(field.grid)

regions(obj) = Tuple(get_region(obj, region_index) for region_index in nregions(obj))

@inline nregions(a, b, c...) = maximum(tuple(a, b, c...))

macro regionalize(expr)
    return quote
        func = expr.args[1]
        func_args = expr.args[2:end]
        number_of_regions = nregions(Tuple($(esc(a)) for a in func_args)...)

        for region_index in 1:number_of_regions
            # Use recursive get_region(t::Tuple, i) method:
            $(esc(func))(get_region($(esc(func_args)), region_index)...)
        end
    end
end
