####
#### Vertical coordinates
####

# This file implements everything related to vertical coordinates in Oceananigans.
# Vertical coordinates are independent of the underlying grid type since only grids that are
# "unstructured" or "curvilinear" in the horizontal directions are supported in Oceananigans.
# Thus the vertical coordinate is _special_, and it can be implemented once for all grid types.

abstract type AbstractVerticalCoordinate end

"""
    struct StaticVerticalDiscretization{C, D, E, F} <: AbstractVerticalCoordinate

Represent a static one-dimensional vertical coordinate.

Fields
======

- `cᶜ::C`: Cell-centered coordinate.
- `cᶠ::D`: Face-centered coordinate.
- `Δᶜ::E`: Cell-centered grid spacing.
- `Δᶠ::F`: Face-centered grid spacing.
"""
struct StaticVerticalDiscretization{C, D, E, F} <: AbstractVerticalCoordinate
    cᵃᵃᶠ :: C
    cᵃᵃᶜ :: D
    Δᵃᵃᶠ :: E
    Δᵃᵃᶜ :: F
end

struct MutableVerticalDiscretization{C, D, E, F, H, CC, FC, CF, FF} <: AbstractVerticalCoordinate
    cᵃᵃᶠ :: C
    cᵃᵃᶜ :: D
    Δᵃᵃᶠ :: E
    Δᵃᵃᶜ :: F
      ηⁿ :: H
    σᶜᶜⁿ :: CC
    σᶠᶜⁿ :: FC
    σᶜᶠⁿ :: CF
    σᶠᶠⁿ :: FF
    σᶜᶜ⁻ :: CC
    ∂t_σ :: CC
end

"""
    MutableVerticalDiscretization(r_faces)

Construct a `MutableVerticalDiscretization` from `r_faces` that can be a `Tuple`, a function of an index `k`,
or an `AbstractArray`. A `MutableVerticalDiscretization` defines a vertical coordinate that might evolve in time
following certain rules. Examples of `MutableVerticalDiscretization`s are free-surface following coordinates,
or sigma coordinates.
"""
MutableVerticalDiscretization(r_faces) = MutableVerticalDiscretization(r_faces, r_faces, (nothing for i in 1:9)...)

####
#### Some useful aliases
####

const RegularStaticVerticalDiscretization  = StaticVerticalDiscretization{<:Any, <:Any, <:Number}
const RegularMutableVerticalDiscretization = MutableVerticalDiscretization{<:Any, <:Any, <:Number}

const RegularVerticalCoordinate = Union{RegularStaticVerticalDiscretization, RegularMutableVerticalDiscretization}

const AbstractMutableGrid = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Bounded, <:MutableVerticalDiscretization}
const AbstractStaticGrid  = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Any,     <:StaticVerticalDiscretization}
const RegularVerticalGrid = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Any,     <:RegularVerticalCoordinate}

####
#### Adapt and on_architecture
####

Adapt.adapt_structure(to, coord::StaticVerticalDiscretization) =
    StaticVerticalDiscretization(Adapt.adapt(to, coord.cᵃᵃᶠ),
                             Adapt.adapt(to, coord.cᵃᵃᶜ),
                             Adapt.adapt(to, coord.Δᵃᵃᶠ),
                             Adapt.adapt(to, coord.Δᵃᵃᶜ))

on_architecture(arch, coord::StaticVerticalDiscretization) =
    StaticVerticalDiscretization(on_architecture(arch, coord.cᵃᵃᶠ),
                             on_architecture(arch, coord.cᵃᵃᶜ),
                             on_architecture(arch, coord.Δᵃᵃᶠ),
                             on_architecture(arch, coord.Δᵃᵃᶜ))

Adapt.adapt_structure(to, coord::MutableVerticalDiscretization) =
    MutableVerticalDiscretization(Adapt.adapt(to, coord.cᵃᵃᶠ),
                              Adapt.adapt(to, coord.cᵃᵃᶜ),
                              Adapt.adapt(to, coord.Δᵃᵃᶠ),
                              Adapt.adapt(to, coord.Δᵃᵃᶜ),
                              Adapt.adapt(to, coord.ηⁿ),
                              Adapt.adapt(to, coord.σᶜᶜⁿ),
                              Adapt.adapt(to, coord.σᶠᶜⁿ),
                              Adapt.adapt(to, coord.σᶜᶠⁿ),
                              Adapt.adapt(to, coord.σᶠᶠⁿ),
                              Adapt.adapt(to, coord.σᶜᶜ⁻),
                              Adapt.adapt(to, coord.∂t_σ))

on_architecture(arch, coord::MutableVerticalDiscretization) =
    MutableVerticalDiscretization(on_architecture(arch, coord.cᵃᵃᶠ),
                              on_architecture(arch, coord.cᵃᵃᶜ),
                              on_architecture(arch, coord.Δᵃᵃᶠ),
                              on_architecture(arch, coord.Δᵃᵃᶜ),
                              on_architecture(arch, coord.ηⁿ),
                              on_architecture(arch, coord.σᶜᶜⁿ),
                              on_architecture(arch, coord.σᶠᶜⁿ),
                              on_architecture(arch, coord.σᶜᶠⁿ),
                              on_architecture(arch, coord.σᶠᶠⁿ),
                              on_architecture(arch, coord.σᶜᶜ⁻),
                              on_architecture(arch, coord.∂t_σ))

#####
##### Nodes and spacings (common to every grid)...
#####

AUG = AbstractUnderlyingGrid

@inline rnode(i, j, k, grid, ℓx, ℓy, ℓz) = rnode(k, grid, ℓz)

@inline function rnode(i::AbstractArray, j::AbstractArray, k, grid, ℓx, ℓy, ℓz)
    res = rnode(k, grid, ℓz)
    toperm = Base.stack(collect(Base.stack(collect(res for _ in 1:size(j, 2))) for _ in 1:size(i, 1)))
    permutedims(toperm, (3, 2, 1))
end
@inline rnode(k, grid, ::Center) = getnode(grid.z.cᵃᵃᶜ, k)
@inline rnode(k, grid, ::Face)   = getnode(grid.z.cᵃᵃᶠ, k)

# These will be extended in the Operators module
@inline znode(k, grid, ℓz) = rnode(k, grid, ℓz)
@inline znode(i, j, k, grid, ℓx, ℓy, ℓz) = rnode(i, j, k, grid, ℓx, ℓy, ℓz)

@inline rnodes(grid::AUG, ℓz::Face;   with_halos=false) = _property(grid.z.cᵃᵃᶠ, ℓz, topology(grid, 3), size(grid, 3), with_halos)
@inline rnodes(grid::AUG, ℓz::Center; with_halos=false) = _property(grid.z.cᵃᵃᶜ, ℓz, topology(grid, 3), size(grid, 3), with_halos)
@inline rnodes(grid::AUG, ℓx, ℓy, ℓz; with_halos=false) = rnodes(grid, ℓz; with_halos)

rnodes(grid::AUG, ::Nothing; kwargs...) = 1:1
znodes(grid::AUG, ::Nothing; kwargs...) = 1:1

# TODO: extend in the Operators module
@inline znodes(grid::AUG, ℓz; kwargs...) = rnodes(grid, ℓz; kwargs...)
@inline znodes(grid::AUG, ℓx, ℓy, ℓz; kwargs...) = rnodes(grid, ℓx, ℓy, ℓz; kwargs...)

function rspacings end
function zspacings end

@inline rspacings(grid, ℓz) = rspacings(grid, nothing, nothing, ℓz)
@inline zspacings(grid, ℓz) = zspacings(grid, nothing, nothing, ℓz)

####
#### `z_domain` and `cpu_face_constructor_z`
####

z_domain(grid) = domain(topology(grid, 3)(), grid.Nz, grid.z.cᵃᵃᶠ)

@inline cpu_face_constructor_r(grid::RegularVerticalGrid) = z_domain(grid)

@inline function cpu_face_constructor_r(grid)
    Nz = size(grid, 3)
    nodes = rnodes(grid, Face(); with_halos=true)
    cpu_nodes = on_architecture(CPU(), nodes)
    return cpu_nodes[1:Nz+1]
end

@inline cpu_face_constructor_z(grid) = cpu_face_constructor_r(grid)
@inline cpu_face_constructor_z(grid::AbstractMutableGrid) = MutableVerticalDiscretization(cpu_face_constructor_r(grid))

####
#### Utilities
####

function validate_dimension_specification(T, ξ::MutableVerticalDiscretization, dir, N, FT)
    cᶠ = validate_dimension_specification(T, ξ.cᵃᵃᶠ, dir, N, FT)
    cᶜ = validate_dimension_specification(T, ξ.cᵃᵃᶜ, dir, N, FT)
    args = Tuple(getproperty(ξ, prop) for prop in propertynames(ξ))

    return MutableVerticalDiscretization(cᶠ, cᶜ, args[3:end]...)
end

# Summaries
coordinate_summary(topo, z::StaticVerticalDiscretization, name) = coordinate_summary(topo, z.Δᵃᵃᶜ, name)

coordinate_summary(::Bounded, z::RegularMutableVerticalDiscretization, name) =
    @sprintf("regularly spaced with Δr=%s (mutable)", prettysummary(z.Δᵃᵃᶜ))

coordinate_summary(::Bounded, z::MutableVerticalDiscretization, name) =
    @sprintf("variably spaced with min(Δr)=%s, max(Δr)=%s (mutable)",
             prettysummary(minimum(z.Δᵃᵃᶜ)),
             prettysummary(maximum(z.Δᵃᵃᶜ)))
