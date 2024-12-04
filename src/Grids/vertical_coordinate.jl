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

####
#### Some usefull aliases
####

const RegularStaticVerticalCoordinate = StaticVerticalCoordinate{<:Any, <:Number}
const RegularVerticalGrid = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Any, <:RegularStaticVerticalCoordinate}

####
#### Adapt and on_architecture
####

Adapt.adapt_structure(to, coord::StaticVerticalCoordinate) = 
   StaticVerticalCoordinate(Adapt.adapt(to, coord.cᶠ),
                            Adapt.adapt(to, coord.cᶜ),
                            Adapt.adapt(to, coord.Δᶠ),
                            Adapt.adapt(to, coord.Δᶜ))

on_architecture(arch, coord::StaticVerticalCoordinate) = 
   StaticVerticalCoordinate(on_architecture(arch, coord.cᶠ),
                            on_architecture(arch, coord.cᶜ),
                            on_architecture(arch, coord.Δᶠ),
                            on_architecture(arch, coord.Δᶜ))

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
#### `z_domain` (independent of ZStar or not) and `cpu_face_constructor_z`
####

z_domain(grid) = domain(topology(grid, 3)(), grid.Nz, grid.z.cᶠ)

@inline cpu_face_constructor_r(grid::RegularVerticalGrid) = z_domain(grid)

@inline function cpu_face_constructor_r(grid)
    Nz = size(grid, 3)
    nodes = rnodes(grid, Face(); with_halos=true)
    cpu_nodes = on_architecture(CPU(), nodes)
    return cpu_nodes[1:Nz+1]
end

@inline cpu_face_constructor_z(grid) = cpu_face_constructor_r(grid)

####
#### Utilities
####

# Summaries
coordinate_summary(::Bounded, z::AbstractVerticalCoordinate, name) = 
    @sprintf("Free-surface following with Δ%s=%s", name, prettysummary(z.Δᶜ))
