using Oceananigans.Fields: validate_field_tuple_grid

#####
##### Forms for IncompressibleModel constructor
#####

DiffusivityFields(diffusivities::NamedTuple, arch, grid, tracer_names, bcs, closure) =
    validate_field_tuple_grid("diffusivities", diffusivities, grid)

DiffusivityFields(::Nothing, arch, grid, tracer_names, bcs, closure) =
    DiffusivityFields(arch, grid, tracer_names, bcs, closure)

#####
##### Closures without precomputed diffusivities
#####

DiffusivityFields(arch, grid, tracer_names, bcs, closure) = nothing

#####
##### Closure tuples
#####

DiffusivityFields(arch, grid, tracer_names, bcs, closure_tuple::Tuple) =
    Tuple(DiffusivityFields(arch, grid, tracer_names, bcs, closure) for closure in closure_tuple)

