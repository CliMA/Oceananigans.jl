# Fallback constructor for diffusivity types without precomputed diffusivities.
DiffusivityFields(arch::AbstractArchitecture, grid::AbstractGrid, args...) = nothing

DiffusivityFields(arch::AbstractArchitecture, grid::AbstractGrid, tracers, closure_tuple::Tuple) =
    Tuple(DiffusivityFields(arch, grid, tracers, closure) for closure in closure_tuple)

#####
##### For closures that only require an eddy viscosity νₑ field.
#####

const NuClosures = Union{AbstractSmagorinsky, AbstractLeith}

DiffusivityFields(
    arch::AbstractArchitecture, grid::AbstractGrid, tracers, ::NuClosures;
    νₑ = CellField(arch, grid, DiffusivityBoundaryConditions(grid), zeros(arch, grid))
    ) = (νₑ=νₑ,)

function DiffusivityFields(arch::AbstractArchitecture, grid::AbstractGrid, tracers,
                           bcs::NamedTuple, ::NuClosures)
    νₑ_bcs = :νₑ ∈ keys(bcs) ? bcs[:νₑ] : DiffusivityBoundaryConditions(grid)
    νₑ = CellField(arch, grid, νₑ_bcs, zeros(arch, grid))
    return (νₑ=νₑ,)
end

#####
##### For closures that also require tracer diffusivity fields κₑ on each tracer.
#####

const NuKappaClosures = Union{VAMD, RAMD}

function DiffusivityFields(
    arch::AbstractArchitecture, grid::AbstractGrid, tracers, ::NuKappaClosures;
    νₑ = CellField(arch, grid, DiffusivityBoundaryConditions(grid), zeros(arch, grid)), kwargs...)
    κₑ = TracerFields(arch, grid, tracers; kwargs...)
    return (νₑ=νₑ, κₑ=κₑ)
end

function DiffusivityFields(arch::AbstractArchitecture, grid::AbstractGrid, tracers,
                           bcs::NamedTuple, ::NuKappaClosures)
    νₑ_bcs = :νₑ ∈ keys(bcs) ? bcs[:νₑ] : DiffusivityBoundaryConditions(grid)
    νₑ = CellField(arch, grid, νₑ_bcs, zeros(arch, grid))
    κₑ = TracerFields(arch, grid, tracers, bcs)
    return (νₑ=νₑ, κₑ=κₑ)
end
