####
#### Vertical coordinates
####

# This file implements everything related to vertical coordinates in Oceananigans.
# Vertical coordinates are independent of the underlying grid type as we support grids that are 
# "unstructured" or "curvilinear" only in the horizontal direction. 
# For this reason the vertical coodinate is _special_, and it can be implemented once for all grid types.

abstract type AbstractVerticalCoordinate end

# Represents a static one-dimensional vertical coordinate.
#
# # Fields
# - `cᶜ::C`: Cell-centered coordinate.
# - `cᶠ::C`: Face-centered coordinate.
# - `Δᶜ::D`: Cell-centered grid spacing.
# - `Δᶠ::D`: Face-centered grid spacing.
struct StaticVerticalCoordinate{C, D} <: AbstractVerticalCoordinate
    cᵃᵃᶠ :: C
    cᵃᵃᶜ :: C
    Δᵃᵃᶠ :: D
    Δᵃᵃᶜ :: D
end

# Represents a z-star three-dimensional vertical coordinate.
#
# # Fields
# - `cᶠ::C`: Face-centered coordinate.
# - `cᶜ::C`: Cell-centered coordinate.
# - `Δᶠ::D`: Face-centered grid spacing.
# - `Δᶜ::D`: Cell-centered grid spacing.
# - `ηⁿ::E`: Surface elevation at the current time step.
# - `σᶜᶜⁿ::CC`: Vertical grid scaling at center-center at the current time step.
# - `σᶠᶜⁿ::FC`: Vertical grid scaling at face-center at the current time step.
# - `σᶜᶠⁿ::CF`: Vertical grid scaling at center-face at the current time step.
# - `σᶠᶠⁿ::FF`: Vertical grid scaling at face-face at the current time step.
# - `σᶜᶜ⁻::CC`: Vertical grid scaling at center-center at the previous time step.
# - `∂t_σ::CC`: Time derivative of the vertical grid scaling at cell centers.
struct ZStarVerticalCoordinate{C, D, E, CC, FC, CF, FF} <: AbstractVerticalCoordinate
    cᵃᵃᶠ :: C
    cᵃᵃᶜ :: C
    Δᵃᵃᶠ :: D
    Δᵃᵃᶜ :: D
      ηⁿ :: E
    σᶜᶜⁿ :: CC
    σᶠᶜⁿ :: FC
    σᶜᶠⁿ :: CF
    σᶠᶠⁿ :: FF
    σᶜᶜ⁻ :: CC
    ∂t_σ :: CC
end

# Convenience constructors for Zstar vertical coordinate
ZStarVerticalCoordinate(r_faces::Union{Function, Tuple, AbstractVector}) = ZStarVerticalCoordinate(r_faces, r_faces, [nothing for i in 1:9]...)
ZStarVerticalCoordinate(r⁻::Number, r⁺::Number) = ZStarVerticalCoordinate((r⁻, r⁺), (r⁻, r⁺), [nothing for i in 1:9]...)

####
#### Some usefull aliases
####

const RegularStaticVerticalCoordinate = StaticVerticalCoordinate{<:Any, <:Number}
const RegularZStarVerticalCoordinate  = ZStarVerticalCoordinate{<:Any,  <:Number}

const RegularVerticalCoordinate = Union{RegularStaticVerticalCoordinate, RegularZStarVerticalCoordinate}

const AbstractZStarGrid   = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Bounded, <:ZStarVerticalCoordinate}
const AbstractStaticGrid  = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Any,     <:StaticVerticalCoordinate}
const RegularVerticalGrid = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Any,     <:RegularVerticalCoordinate}

####
#### Adapt and on_architecture
####

Adapt.adapt_structure(to, coord::StaticVerticalCoordinate) = 
   StaticVerticalCoordinate(Adapt.adapt(to, coord.cᵃᵃᶠ),
                            Adapt.adapt(to, coord.cᵃᵃᶜ),
                            Adapt.adapt(to, coord.Δᵃᵃᶠ),
                            Adapt.adapt(to, coord.Δᵃᵃᶜ))

on_architecture(arch, coord::StaticVerticalCoordinate) = 
   StaticVerticalCoordinate(on_architecture(arch, coord.cᵃᵃᶠ),
                            on_architecture(arch, coord.cᵃᵃᶜ),
                            on_architecture(arch, coord.Δᵃᵃᶠ),
                            on_architecture(arch, coord.Δᵃᵃᶜ))

Adapt.adapt_structure(to, coord::ZStarVerticalCoordinate) = 
    ZStarVerticalCoordinate(Adapt.adapt(to, coord.cᵃᵃᶠ),
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

on_architecture(arch, coord::ZStarVerticalCoordinate) = 
    ZStarVerticalCoordinate(on_architecture(arch, coord.cᵃᵃᶠ),
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
@inline cpu_face_constructor_z(grid::AbstractZStarGrid) = ZStarVerticalCoordinate(cpu_face_constructor_r(grid))

####
#### Utilities
####

function validate_dimension_specification(T, ξ::ZStarVerticalCoordinate, dir, N, FT)
    cᶠ = validate_dimension_specification(T, ξ.cᵃᵃᶠ, dir, N, FT)
    cᶜ = validate_dimension_specification(T, ξ.cᵃᵃᶜ, dir, N, FT)
    args = Tuple(getproperty(ξ, prop) for prop in propertynames(ξ))

    return ZStarVerticalCoordinate(cᶠ, cᶜ, args[3:end]...)
end

# Summaries
coordinate_summary(topo, z::StaticVerticalCoordinate, name) = coordinate_summary(topo, z.Δᵃᵃᶜ, name)

coordinate_summary(::Bounded, z::RegularZStarVerticalCoordinate, name) = 
    @sprintf("Free-surface following with Δr=%s", prettysummary(z.Δᵃᵃᶜ))

coordinate_summary(::Bounded, z::ZStarVerticalCoordinate, name) = 
    @sprintf("Free-surface following with min(Δr)=%s, max(Δr)=%s", 
             prettysummary(minimum(z.Δᵃᵃᶜ)), 
             prettysummary(maximum(z.Δᵃᵃᶜ)))
