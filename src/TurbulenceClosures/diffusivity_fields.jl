using Oceananigans.Fields: validate_field_tuple_grid

#####
##### Forms for NonhydrostaticModel constructor
#####

function build_diffusivity_fields(diffusivity_fields::NamedTuple, grid, clock, tracer_names, bcs, closure)
    validate_field_tuple_grid("diffusivity_fields", diffusivity_fields, grid)

    return diffusivity_fields
end

build_diffusivity_fields(::Nothing, grid, clock, tracer_names, bcs, closure) =
    build_diffusivity_fields(grid, clock, tracer_names, bcs, closure)

#####
##### Closures without precomputed diffusivities
#####

build_diffusivity_fields(grid, clock, tracer_names, bcs, closure) = nothing

#####
##### Closure tuples
#####

build_diffusivity_fields(grid, clock, tracer_names, bcs, closure_tuple::Tuple) =
    Tuple(build_diffusivity_fields(grid, clock, tracer_names, bcs, closure) for closure in closure_tuple)

