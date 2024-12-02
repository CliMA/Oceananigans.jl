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
    cᶠ :: C
    cᶜ :: C
    Δᶠ :: D
    Δᶜ :: D
end

# Represents a z-star three-dimensional vertical coordinate.
#
# # Fields
# - `cᶜ::C`: Cell-centered coordinate.
# - `cᶠ::C`: Face-centered coordinate.
# - `Δᶜ::D`: Cell-centered grid spacing.
# - `Δᶠ::D`: Face-centered grid spacing.
# - `ηⁿ::E`: Surface elevation at the current time step.
# - `e₃ᶜᶜⁿ::CC`: Vertical grid scaling at center-center at the current time step.
# - `e₃ᶠᶜⁿ::FC`: Vertical grid scaling at face-center at the current time step.
# - `e₃ᶜᶠⁿ::CF`: Vertical grid scaling at center-face at the current time step.
# - `e₃ᶠᶠⁿ::FF`: Vertical grid scaling at face-face at the current time step.
# - `e₃ᶜᶜ⁻::CC`: Vertical grid scaling at center-center at the previous time step.
# - `∂t_e₃::CC`: Time derivative of the vertical grid scaling at cell centers.
struct ZStarVerticalCoordinate{C, D, E, CC, FC, CF, FF} <: AbstractVerticalCoordinate
       cᶠ :: C
       cᶜ :: C
       Δᶠ :: D
       Δᶜ :: D
       ηⁿ :: E
    e₃ᶜᶜⁿ :: CC
    e₃ᶠᶜⁿ :: FC
    e₃ᶜᶠⁿ :: CF
    e₃ᶠᶠⁿ :: FF
    e₃ᶜᶜ⁻ :: CC
    ∂t_e₃ :: CC
end

# Convenience constructors for Zstar vertical coordinate
ZStarVerticalCoordinate(r_faces::Union{Tuple, AbstractVector}) = ZStarVerticalCoordinate(r_faces, r_faces, [nothing for i in 1:9]...)
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
   StaticVerticalCoordinate(Adapt.adapt(to, coord.cᶠ),
                            Adapt.adapt(to, coord.cᶜ),
                            Adapt.adapt(to, coord.Δᶠ),
                            Adapt.adapt(to, coord.Δᶜ))

Adapt.adapt_structure(to, coord::ZStarVerticalCoordinate) = 
    ZStarVerticalCoordinate(Adapt.adapt(to, coord.cᶠ),
                            Adapt.adapt(to, coord.cᶜ),
                            Adapt.adapt(to, coord.Δᶠ),
                            Adapt.adapt(to, coord.Δᶜ),
                            Adapt.adapt(to, coord.ηⁿ),
                            Adapt.adapt(to, coord.e₃ᶜᶜⁿ),
                            Adapt.adapt(to, coord.e₃ᶠᶜⁿ),
                            Adapt.adapt(to, coord.e₃ᶜᶠⁿ),
                            Adapt.adapt(to, coord.e₃ᶠᶠⁿ),
                            Adapt.adapt(to, coord.e₃ᶜᶜ⁻),
                            Adapt.adapt(to, coord.∂t_e₃))

on_architecture(arch, coord::StaticVerticalCoordinate) = 
   StaticVerticalCoordinate(on_architecture(arch, coord.cᶠ),
                            on_architecture(arch, coord.cᶜ),
                            on_architecture(arch, coord.Δᶠ),
                            on_architecture(arch, coord.Δᶜ))

on_architecture(arch, coord::ZStarVerticalCoordinate) = 
    ZStarVerticalCoordinate(on_architecture(arch, coord.cᶠ),
                            on_architecture(arch, coord.cᶜ),
                            on_architecture(arch, coord.Δᶠ),
                            on_architecture(arch, coord.Δᶜ),
                            on_architecture(arch, coord.ηⁿ),
                            on_architecture(arch, coord.e₃ᶜᶜⁿ),
                            on_architecture(arch, coord.e₃ᶠᶜⁿ),
                            on_architecture(arch, coord.e₃ᶜᶠⁿ),
                            on_architecture(arch, coord.e₃ᶠᶠⁿ),
                            on_architecture(arch, coord.e₃ᶜᶜ⁻),
                            on_architecture(arch, coord.∂t_e₃))
 
#####
##### Nodes and spacings (common to every grid)...
#####

AUG = AbstractUnderlyingGrid

@inline rnode(i, j, k, grid, ℓx, ℓy, ℓz) = rnode(k, grid, ℓz)
@inline rnode(k, grid, ::Center) = getnode(grid.z.cᶜ, k)
@inline rnode(k, grid, ::Face)   = getnode(grid.z.cᶠ, k)

# These will be extended in the Operators module
@inline znode(k, grid, ℓz) = rnode(k, grid, ℓz)
@inline znode(i, j, k, grid, ℓx, ℓy, ℓz) = rnode(i, j, k, grid, ℓx, ℓy, ℓz)

@inline rnodes(grid::AUG, ℓz::Face;   with_halos=false) = _property(grid.z.cᶠ, ℓz, topology(grid, 3), size(grid, 3), with_halos)
@inline rnodes(grid::AUG, ℓz::Center; with_halos=false) = _property(grid.z.cᶜ, ℓz, topology(grid, 3), size(grid, 3), with_halos)
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
#### z_domain (independent of ZStar or not)
####

z_domain(grid) = domain(topology(grid, 3)(), grid.Nz, grid.z.cᶠ)

@inline cpu_face_constructor_r(grid::RegularVerticalGrid) = z_domain(grid)
@inline cpu_face_constructor_r(grid) = on_architecture(CPU(), rnodes(grid, Face()))

@inline cpu_face_constructor_z(grid) = cpu_face_constructor_r(grid)
@inline cpu_face_constructor_z(grid::AbstractZStarGrid) = ZStarVerticalCoordinate(cpu_face_constructor_r(grid))

####
#### Utilities
####

function validate_dimension_specification(T, ξ::ZStarVerticalCoordinate, dir, N, FT)
    cᶠ = validate_dimension_specification(T, ξ.cᶠ, dir, N, FT)
    cᶜ = validate_dimension_specification(T, ξ.cᶜ, dir, N, FT)
    args = Tuple(getproperty(ξ, prop) for prop in propertynames(ξ))

    return ZStarVerticalCoordinate(cᶠ, cᶜ, args[3:end]...)
end

# Summaries
coordinate_summary(::Bounded, z::AbstractVerticalCoordinate, name) = 
    @sprintf("Free-surface following with Δ%s=%s", name, prettysummary(z.Δᶜ))
