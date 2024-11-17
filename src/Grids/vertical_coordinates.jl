abstract type AbstractVerticalCoordinate end

struct StaticVerticalCoordinate{C, D} <: AbstractVerticalCoordinate
    cᶠ :: C
    cᶜ :: C
    Δᶠ :: D
    Δᶠ :: D
end

struct ZStarVerticalCoordinate{C, D, E, S} <: AbstractVerticalCoordinate
    cᶠ :: C
    cᶜ :: C
    Δᶠ :: D
    Δᶠ :: D
    ηⁿ :: E
    η⁻ :: E
  ∂t_s :: S
end

const AbstractZStarGrid = AbstractGrid{<:Any, <:Any, <:Any, <:Bounded, <:ZStarVerticalCoordinate}

#####
##### vertical nodes...
#####

@inline rnode(i, j, k, grid, ℓx, ℓy, ℓz) = rnode(k, grid, ℓz)
@inline rnode(k, grid, ::Center) = getnode(grid.z.cᶜ, k)
@inline rnode(k, grid, ::Face)   = getnode(grid.z.cᶠ, k)

@inline znode(k, grid, ℓz)               = rnode(k, grid, ℓz)
@inline znode(i, j, k, grid, ℓx, ℓy, ℓz) = rnode(k, grid, ℓz)

# TO extend in operators
@inline znode(i, j, k, grid::AbstractZStarGrid, ℓx, ℓy, ℓz) = znode(k, grid, ℓz)

@inline znode(k, grid, ℓx)       = getnode(grid.z, k)

# Convenience constructors
ZStarVerticalCoordinate(r_faces::Union{Tuple, AbstractVector}) = ZStarVerticalCoordinate(r_faces, r_faces, nothing, nothing, nothing, nothing, nothing)
ZStarVerticalCoordinate(r⁻::Number, r⁺::Number) = ZStarVerticalCoordinate((r⁻, r⁺), (r⁻, r⁺), nothing, nothing, nothing, nothing, nothing)

Grids.coordinate_summary(::Bounded, Δ::ZStarVerticalCoordinate, name) = 
    @sprintf("Free-surface following with Δ%s=%s", name, prettysummary(Δ.reference))

Grids.coordinate_summary(::Bounded, Δ::ZStarVerticalCoordinate, name) = 
    @sprintf("Free-surface following with Δ%s=%s", name, prettysummary(Δ.reference))

function validate_dimension_specification(T, ξ::ZStarVerticalCoordinate, dir, N, FT)
    reference = validate_dimension_specification(T, ξ.reference, dir, N, FT)
    args      = Tuple(getproperty(ξ, prop) for prop in propertynames(ξ))

    return ZStarVerticalCoordinate(reference, args[2:end]...)
end
