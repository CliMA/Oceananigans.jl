using Oceananigans.Fields: validate_field_tuple_grid

#####
##### Forms for NonhydrostaticModel constructor
#####

function DiffusivityFields(diffusivity_fields::NamedTuple, grid, tracer_names, bcs, closure)
    validate_field_tuple_grid("diffusivity_fields", diffusivity_fields, grid)
    
    return diffusivity_fields
end

DiffusivityFields(::Nothing, grid, tracer_names, bcs, closure) =
    DiffusivityFields(grid, tracer_names, bcs, closure)

#####
##### Closures without precomputed diffusivities
#####

DiffusivityFields(grid, tracer_names, bcs, closure) = nothing

#####
##### Closure tuples
#####

DiffusivityFields(grid, tracer_names, bcs, closure_tuple::Tuple) =
    Tuple(DiffusivityFields(grid, tracer_names, bcs, closure) for closure in closure_tuple)

