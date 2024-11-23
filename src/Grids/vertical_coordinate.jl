abstract type AbstractVerticalCoordinate end

# Represents a static vertical coordinate system.
#
# # Fields
# - `cᶠ::C`: Face-centered coordinate.
# - `cᶜ::C`: Cell-centered coordinate.
# - `Δᶠ::D`: Face-centered grid spacing.
# - `Δᶠ::D`: Face-centered grid spacing (duplicate field, consider renaming or removing).
#
# # Type Parameters
# - `C`: Type of the face-centered and cell-centered coordinates.
# - `D`: Type of the face-centered grid spacing.
struct StaticVerticalCoordinate{C, D} <: AbstractVerticalCoordinate
    cᶠ :: C
    cᶜ :: C
    Δᶜ :: D
    Δᶠ :: D
end

# Represents a z-star vertical coordinate system.
#
# # Fields
# - `cᶠ::C`: Face-centered coordinates.
# - `cᶜ::C`: Cell-centered coordinates.
# - `Δᶠ::D`: Face-centered grid spacing.
# - `Δᶠ::D`: Face-centered grid spacing (duplicate field, consider renaming or removing).
# - `ηⁿ::E`: Surface elevation at the current time step.
# - `η⁻::E`: Surface elevation at the previous time step.
# - `e₃ᶜᶜⁿ::CC`: Vertical grid scaling at cell centers at the current time step.
# - `e₃ᶠᶜⁿ::FC`: Vertical grid scaling at face centers at the current time step.
# - `e₃ᶜᶠⁿ::CF`: Vertical grid scaling at cell-face interfaces at the current time step.
# - `e₃ᶠᶠⁿ::FF`: Vertical grid scaling at face-face interfaces at the current time step.
# - `e₃ᶜᶜ⁻::CC`: Vertical grid scaling at cell centers at the previous time step.
# - `∂t_e₃::CC`: Time derivative of the vertical grid scaling at cell centers.
struct ZStarVerticalCoordinate{C, D, E, CC, FC, CF, FF} <: AbstractVerticalCoordinate
       cᶠ :: C
       cᶜ :: C
       Δᶜ :: D
       Δᶠ :: D
       ηⁿ :: E
    e₃ᶜᶜⁿ :: CC
    e₃ᶠᶜⁿ :: FC
    e₃ᶜᶠⁿ :: CF
    e₃ᶠᶠⁿ :: FF
    e₃ᶜᶜ⁻ :: CC
    ∂t_e₃ :: CC
end

# Convenience constructors for Zstar vertical coordinate
ZStarVerticalCoordinate(r_faces::Union{Tuple, AbstractVector}) = ZStarVerticalCoordinate(r_faces, r_faces, nothing, nothing, nothing, nothing, nothing)
ZStarVerticalCoordinate(r⁻::Number, r⁺::Number) = ZStarVerticalCoordinate((r⁻, r⁺), (r⁻, r⁺), nothing, nothing, nothing, nothing, nothing)

####
#### Some usefull aliases
####

const RegularStaticVerticalCoordinate = StaticVerticalCoordinate{<:Any, <:Number}
const RegularZstarVerticalCoordinate  = ZStarVerticalCoordinate{<:Any,  <:Number}

const RegularVerticalCoordinate = Union{RegularStaticVerticalCoordinate, RegularZstarVerticalCoordinate}

const AbstractZStarGrid   = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Bounded, <:ZStarVerticalCoordinate}
const AbstractStaticGrid  = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Any,     <:StaticVerticalCoordinate}
const RegularVerticalGrid = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Any,     <:RegularVerticalCoordinate}

####
#### Adapting
####

Adapt.adapt_structure(to, coord::StaticVerticalCoordinate) = 
   StaticVerticalCoordinate(Adapt.adapt(to, coord.cᶠ),
                            Adapt.adapt(to, coord.cᶜ),
                            Adapt.adapt(to, coord.Δᶠ),
                            Adapt.adapt(to, coord.Δᶠ))

Adapt.adapt_structure(to, coord::ZStarVerticalCoordinate) = 
    ZStarVerticalCoordinate(Adapt.adapt(to, coord.cᶠ),
                            Adapt.adapt(to, coord.cᶜ),
                            Adapt.adapt(to, coord.Δᶠ),
                            Adapt.adapt(to, coord.Δᶠ),
                            Adapt.adapt(to, coord.ηⁿ),
                            Adapt.adapt(to, coord.e₃ᶜᶜⁿ),
                            Adapt.adapt(to, coord.e₃ᶠᶜⁿ),
                            Adapt.adapt(to, coord.e₃ᶜᶠⁿ),
                            Adapt.adapt(to, coord.e₃ᶠᶠⁿ),
                            Adapt.adapt(to, coord.e₃ᶜᶜ⁻),
                            Adapt.adapt(to, coord.∂t_e₃))

#####
##### Vertical nodes...
#####

@inline rnode(i, j, k, grid, ℓx, ℓy, ℓz) = rnode(k, grid, ℓz)
@inline rnode(k, grid, ::Center) = getnode(grid.z.cᶜ, k)
@inline rnode(k, grid, ::Face) = getnode(grid.z.cᶠ, k)

@inline znode(k, grid, ℓz) = rnode(k, grid, ℓz)
@inline znode(i, j, k, grid, ℓx, ℓy, ℓz) = rnode(k, grid, ℓz)

# Extended for ZStarGrids in the Operators module
@inline znode(i, j, k, grid, ℓx, ℓy, ℓz) = znode(k, grid, ℓz)
@inline znode(k, grid, ℓx) = getnode(grid.z, k)

# Summaries
Grids.coordinate_summary(::Bounded, z::ZStarVerticalCoordinate, name) = 
    @sprintf("Free-surface following with Δ%s=%s", name, prettysummary(z.Δᶜ))

function validate_dimension_specification(T, ξ::ZStarVerticalCoordinate, dir, N, FT)
    reference = validate_dimension_specification(T, ξ.reference, dir, N, FT)
    args      = Tuple(getproperty(ξ, prop) for prop in propertynames(ξ))

    return ZStarVerticalCoordinate(reference, args[2:end]...)
end
