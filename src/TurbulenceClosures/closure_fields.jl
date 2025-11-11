using Oceananigans.Fields: validate_field_tuple_grid

#####
##### Forms for NonhydrostaticModel constructor
#####

function build_closure_fields(closure_fields::NamedTuple, grid, clock, tracer_names, bcs, closure)
    validate_field_tuple_grid("closure_fields", closure_fields, grid)

    return closure_fields
end

build_closure_fields(::Nothing, grid, clock, tracer_names, bcs, closure) =
    build_closure_fields(grid, clock, tracer_names, bcs, closure)

#####
##### Closures without precomputed diffusivities
#####

build_closure_fields(grid, clock, tracer_names, bcs, closure) = nothing

#####
##### Closure tuples
#####

build_closure_fields(grid, clock, tracer_names, bcs, closure_tuple::Tuple) =
    Tuple(build_closure_fields(grid, clock, tracer_names, bcs, closure) for closure in closure_tuple)

