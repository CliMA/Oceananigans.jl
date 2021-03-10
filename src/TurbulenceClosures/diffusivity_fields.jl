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

#####
##### For closures that only require an eddy viscosity νₑ field.
#####

const ViscosityClosures = Union{SmagorinskyLilly, AbstractLeith}

DiffusivityFields(arch, grid, tracer_names, ::ViscosityClosures;
                  νₑ = CenterField(arch, grid, DiffusivityBoundaryConditions(grid))) = (νₑ=νₑ,)

function DiffusivityFields(arch, grid, tracer_names, bcs, closure::ViscosityClosures)
    νₑ_bcs = :νₑ ∈ keys(bcs) ? bcs[:νₑ] : DiffusivityBoundaryConditions(grid)
    return DiffusivityFields(arch, grid, tracer_names, closure; νₑ = CenterField(arch, grid, νₑ_bcs))
end

#####
##### For closures that also require tracer diffusivity fields κₑ on each tracer.
#####

function DiffusivityFields(arch, grid, tracer_names, ::AMD;
                           νₑ = CenterField(arch, grid, DiffusivityBoundaryConditions(grid)),
                           kwargs...)

    κₑ = TracerDiffusivityFields(arch, grid, tracer_names; kwargs...)

    return (νₑ=νₑ, κₑ=κₑ)
end

function DiffusivityFields(arch, grid, tracer_names, bcs, ::AMD)

    νₑ_bcs = :νₑ ∈ keys(bcs) ? bcs[:νₑ] : DiffusivityBoundaryConditions(grid)
    νₑ = CenterField(arch, grid, νₑ_bcs)

    κₑ = :κₑ ∈ keys(bcs) ? TracerDiffusivityFields(arch, grid, tracer_names, bcs[:κₑ]) :
                           TracerDiffusivityFields(arch, grid, tracer_names)

    return (νₑ=νₑ, κₑ=κₑ)
end

function TracerDiffusivityFields(arch, grid, tracer_names; kwargs...)

    κ_fields = Tuple(c ∈ keys(kwargs) ?
                     kwargs[c] :
                     CenterField(arch, grid, DiffusivityBoundaryConditions(grid))
                     for c in tracer_names)

    return NamedTuple{tracer_names}(κ_fields)
end

function TracerDiffusivityFields(arch, grid, tracer_names, bcs)

    κ_fields = Tuple(c ∈ keys(bcs) ? CenterField(arch, grid, bcs[c]) :
                                     CenterField(arch, grid, DiffusivityBoundaryConditions(grid))
                     for c in tracer_names)

    return NamedTuple{tracer_names}(κ_fields)
end
