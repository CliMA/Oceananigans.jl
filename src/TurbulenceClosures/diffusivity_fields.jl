# Fallback constructor for diffusivity types without precomputed diffusivities.
DiffusivityFields(arch::AbstractArchitecture, grid::AbstractGrid, args...) = nothing

# For a tuple of closures
DiffusivityFields(
    arch::AbstractArchitecture, grid::AbstractGrid, tracers,
    bcs::NamedTuple, closure_tuple::Tuple
) = Tuple(DiffusivityFields(arch, grid, tracers, bcs, closure) for closure in closure_tuple)

#####
##### For closures that only require an eddy viscosity νₑ field.
#####

const ViscosityClosures = Union{AbstractSmagorinsky, AbstractLeith}

DiffusivityFields(
    arch::AbstractArchitecture, grid::AbstractGrid, tracers, ::ViscosityClosures;
    νₑ = CellField(arch, grid, DiffusivityBoundaryConditions(grid), zeros(arch, grid))
) = (νₑ=νₑ,)

function DiffusivityFields(arch::AbstractArchitecture, grid::AbstractGrid, tracers,
                           bcs::NamedTuple, ::ViscosityClosures)
    νₑ_bcs = :νₑ ∈ keys(bcs) ? bcs[:νₑ] : DiffusivityBoundaryConditions(grid)
    νₑ = CellField(arch, grid, νₑ_bcs, zeros(arch, grid))
    return (νₑ=νₑ,)
end

#####
##### For closures that also require tracer diffusivity fields κₑ on each tracer.
#####

const ViscosityDiffusivityClosures = Union{VAMD, RAMD}

function TracerDiffusivityFields(arch, grid, tracer_names; kwargs...)
    κ_fields =
        Tuple(c ∈ keys(kwargs) ?
              kwargs[c] :
              CellField(arch, grid, DiffusivityBoundaryConditions(grid), zeros(arch, grid))
              for c in tracer_names)
    return NamedTuple{tracer_names}(κ_fields)
end

function TracerDiffusivityFields(arch, grid, tracer_names, bcs::NamedTuple)
    κ_fields =
        Tuple(c ∈ keys(bcs) ?
              CellField(arch, grid, bcs[c],                              zeros(arch, grid)) :
              CellField(arch, grid, DiffusivityBoundaryConditions(grid), zeros(arch, grid))
              for c in tracer_names)
    return NamedTuple{tracer_names}(κ_fields)
end

function DiffusivityFields(
    arch::AbstractArchitecture, grid::AbstractGrid, tracers, ::ViscosityDiffusivityClosures;
    νₑ = CellField(arch, grid, DiffusivityBoundaryConditions(grid), zeros(arch, grid)), kwargs...)
    κₑ = TracerDiffusivityFields(arch, grid, tracers; kwargs...)
    return (νₑ=νₑ, κₑ=κₑ)
end

function DiffusivityFields(arch::AbstractArchitecture, grid::AbstractGrid, tracers,
                           bcs::NamedTuple, ::ViscosityDiffusivityClosures)

    νₑ_bcs = :νₑ ∈ keys(bcs) ? bcs[:νₑ] : DiffusivityBoundaryConditions(grid)
    νₑ = CellField(arch, grid, νₑ_bcs, zeros(arch, grid))

    κₑ = :κₑ ∈ keys(bcs) ?
        TracerDiffusivityFields(arch, grid, tracers, bcs[:κₑ]) :
        TracerDiffusivityFields(arch, grid, tracers)

    return (νₑ=νₑ, κₑ=κₑ)
end
