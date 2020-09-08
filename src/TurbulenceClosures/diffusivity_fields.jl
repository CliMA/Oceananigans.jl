# Fallback constructor for diffusivity types without precomputed diffusivities.
DiffusivityFields(arch, grid, tracer_names, bcs, closure) = nothing
DiffusivityFields(arch, grid, tracer_names, closure) = nothing

# For a tuple of closures
DiffusivityFields(arch, grid, tracer_names, bcs, closure_tuple::Tuple) =
    Tuple(DiffusivityFields(arch, grid, tracer_names, bcs, closure) for closure in closure_tuple)

#####
##### For closures that only require an eddy viscosity νₑ field.
#####

const ViscosityClosures = Union{AbstractSmagorinsky, AbstractLeith}

DiffusivityFields(arch, grid, tracer_names, ::ViscosityClosures;
                  νₑ = CellField(arch, grid, DiffusivityBoundaryConditions(grid))) = (νₑ=νₑ,)

function DiffusivityFields(arch, grid, tracer_names, bcs, ::ViscosityClosures)

    νₑ_bcs = :νₑ ∈ keys(bcs) ? bcs[:νₑ] : DiffusivityBoundaryConditions(grid)

    νₑ = CellField(arch, grid, νₑ_bcs)

    return (νₑ=νₑ,)
end

#####
##### For closures that also require tracer diffusivity fields κₑ on each tracer.
#####

const ViscosityDiffusivityClosures = Union{VAMD, RAMD}

function TracerDiffusivityFields(arch, grid, tracer_names; kwargs...)

    κ_fields = Tuple(c ∈ keys(kwargs) ?
                     kwargs[c] :
                     CellField(arch, grid, DiffusivityBoundaryConditions(grid))
                     for c in tracer_names)

    return NamedTuple{tracer_names}(κ_fields)
end

function TracerDiffusivityFields(arch, grid, tracer_names, bcs)

    κ_fields = Tuple(c ∈ keys(bcs) ? CellField(arch, grid, bcs[c]) :
                                     CellField(arch, grid, DiffusivityBoundaryConditions(grid))
                     for c in tracer_names)

    return NamedTuple{tracer_names}(κ_fields)
end

function DiffusivityFields(arch, grid, tracer_names, ::ViscosityDiffusivityClosures;
                           νₑ = CellField(arch, grid, DiffusivityBoundaryConditions(grid)),
                           kwargs...)

    κₑ = TracerDiffusivityFields(arch, grid, tracer_names; kwargs...)

    return (νₑ=νₑ, κₑ=κₑ)
end

function DiffusivityFields(arch, grid, tracer_names, bcs, ::ViscosityDiffusivityClosures)

    νₑ_bcs = :νₑ ∈ keys(bcs) ? bcs[:νₑ] : DiffusivityBoundaryConditions(grid)

    νₑ = CellField(arch, grid, νₑ_bcs)

    κₑ = :κₑ ∈ keys(bcs) ? TracerDiffusivityFields(arch, grid, tracer_names, bcs[:κₑ]) :
                           TracerDiffusivityFields(arch, grid, tracer_names)

    return (νₑ=νₑ, κₑ=κₑ)
end
