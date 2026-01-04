####
#### Vertical coordinates
####

# This file implements everything related to vertical coordinates in Oceananigans.
# Vertical coordinates are independent of the underlying grid type since only grids that are
# "unstructured" or "curvilinear" in the horizontal directions are supported in Oceananigans.
# Thus the vertical coordinate is _special_, and it can be implemented once for all grid types.
#
# Notation:
#   - (ξ, η, r) are computational coordinates
#   - (x, y, z) are physical coordinates
#   - η_fs (stored as ηⁿ in code) is the free surface displacement
#   - σ = ∂z/∂r is the specific thickness (stretching factor)

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

# Summaries
const RegularStaticVerticalDiscretization  = StaticVerticalDiscretization{<:Any, <:Any, <:Number}
const AbstractStaticGrid  = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Any, <:StaticVerticalDiscretization}

coordinate_summary(topo, z::StaticVerticalDiscretization, name) = coordinate_summary(topo, z.Δᵃᵃᶜ, name)

#####
##### GeneralizedVerticalDiscretization - user-facing input type
#####

"""
    struct GeneralizedVerticalDiscretization{R}

A user-facing type for specifying a generalized (potentially time-varying) vertical coordinate.

This type is used as input when constructing a grid. It specifies the reference vertical
coordinate interfaces `r_faces`. When the grid is constructed, this is converted to a
`ZStarVerticalCoordinate` which stores both the reference coordinate and the time-evolving
state needed for zee-star or other generalized vertical coordinates.

See [Generalized vertical coordinates](@ref generalized_vertical_coordinates) for the theory.

# Example

```julia
using Oceananigans

# Create a grid with a generalized vertical discretization
z = GeneralizedVerticalDiscretization((-100, 0))
grid = RectilinearGrid(size=(10, 10, 10), x=(0, 1), y=(0, 1), z=z)
```
"""
struct GeneralizedVerticalDiscretization{R}
    r_faces :: R
end

Base.show(io::IO, gvd::GeneralizedVerticalDiscretization) =
    print(io, "GeneralizedVerticalDiscretization with reference interfaces: ", gvd.r_faces)

#####
##### ZStarVerticalCoordinate - grid-stored type with time-evolving state
#####

"""
    struct ZStarVerticalCoordinate{C, D, E, F, H, CC, FC, CF, FF} <: AbstractVerticalCoordinate

A generalized vertical coordinate stored on the grid, supporting time-varying coordinates
such as the zee-star (z*) free-surface-following coordinate.

This type stores both the reference vertical coordinate (r) and the time-evolving state
needed for coordinate transformations.

# Fields

Reference coordinate (static):
- `cᵃᵃᶠ`: Face-centered reference coordinate values
- `cᵃᵃᶜ`: Cell-centered reference coordinate values
- `Δᵃᵃᶠ`: Face-centered reference coordinate spacings
- `Δᵃᵃᶜ`: Cell-centered reference coordinate spacings

Time-evolving state:
- `ηⁿ`: Free surface displacement at current time (denoted η_fs in docs)
- `σᶜᶜⁿ`: Specific thickness σ = ∂z/∂r at (Center, Center) at current time
- `σᶠᶜⁿ`: Specific thickness at (Face, Center) at current time
- `σᶜᶠⁿ`: Specific thickness at (Center, Face) at current time
- `σᶠᶠⁿ`: Specific thickness at (Face, Face) at current time
- `σᶜᶜ⁻`: Specific thickness at (Center, Center) at previous time
- `∂t_σ`: Time derivative of specific thickness at (Center, Center)

See [Generalized vertical coordinates](@ref generalized_vertical_coordinates) for the theory.
"""
struct ZStarVerticalCoordinate{C, D, E, F, H, CC, FC, CF, FF} <: AbstractVerticalCoordinate
    # Reference coordinate (static)
    cᵃᵃᶠ :: C
    cᵃᵃᶜ :: D
    Δᵃᵃᶠ :: E
    Δᵃᵃᶜ :: F
    # Time-evolving state
      ηⁿ :: H   # Free surface displacement (η_fs in docs notation)
    σᶜᶜⁿ :: CC  # Specific thickness σ = ∂z/∂r at various staggerings
    σᶠᶜⁿ :: FC
    σᶜᶠⁿ :: CF
    σᶠᶠⁿ :: FF
    σᶜᶜ⁻ :: CC  # Previous time level
    ∂t_σ :: CC  # Time derivative of σ
end

####
#### Some useful aliases
####

const RegularZStarVerticalCoordinate = ZStarVerticalCoordinate{<:Any, <:Any, <:Number}
const RegularVerticalCoordinate = Union{RegularStaticVerticalDiscretization, RegularZStarVerticalCoordinate}

const AbstractGeneralizedVerticalGrid = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Bounded, <:ZStarVerticalCoordinate}
const RegularVerticalGrid = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Any, <:RegularVerticalCoordinate}

# Backward compatibility aliases
const MutableVerticalDiscretization = GeneralizedVerticalDiscretization
const RegularMutableVerticalDiscretization = RegularZStarVerticalCoordinate
const AbstractMutableGrid = AbstractGeneralizedVerticalGrid

coordinate_summary(::Bounded, z::RegularZStarVerticalCoordinate, name) =
    @sprintf("regularly spaced with generalized Δr=%s", prettysummary(z.Δᵃᵃᶜ))

coordinate_summary(::Bounded, z::ZStarVerticalCoordinate, name) =
    @sprintf("variably spaced generalized coordinate with min(Δr)=%s, max(Δr)=%s",
             prettysummary(minimum(parent(z.Δᵃᵃᶜ))),
             prettysummary(maximum(parent(z.Δᵃᵃᶜ))))

function Base.show(io::IO, z::ZStarVerticalCoordinate)
    print(io, "ZStarVerticalCoordinate with reference interfaces r:\n")
    Base.show(io, z.cᵃᵃᶠ)
end

#####
##### Coordinate generation for grid constructors
#####

generate_coordinate(FT, ::Periodic, N, H, ::GeneralizedVerticalDiscretization, coordinate_name, arch, args...) =
    throw(ArgumentError("Periodic domains are not supported for GeneralizedVerticalDiscretization"))

# Generate a vertical coordinate with a scaling (`σ`) with respect to a reference coordinate `r` with spacing `Δr`.
# The grid might move with time, so the coordinate includes the time-derivative of the scaling `∂t_σ`.
# The value of the vertical coordinate at `Nz+1` is saved in `ηⁿ`.
function generate_coordinate(FT, topo, size, halo, coordinate::GeneralizedVerticalDiscretization, coordinate_name, dim::Int, arch)

    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo

    if dim != 3
        msg = "GeneralizedVerticalDiscretization is supported only in the third dimension (z)"
        throw(ArgumentError(msg))
    end

    if coordinate_name != :z
        msg = "GeneralizedVerticalDiscretization is supported only for the z-coordinate"
        throw(ArgumentError(msg))
    end

    r_faces = coordinate.r_faces

    LR, rᵃᵃᶠ, rᵃᵃᶜ, Δrᵃᵃᶠ, Δrᵃᵃᶜ = generate_coordinate(FT, topo[3](), Nz, Hz, r_faces, :r, arch)

    args = (topo, (Nx, Ny, Nz), (Hx, Hy, Hz))

    # Allocate time-evolving state arrays
    σᶜᶜ⁻ = new_data(FT, arch, (Center, Center, Nothing), args...)
    σᶜᶜⁿ = new_data(FT, arch, (Center, Center, Nothing), args...)
    σᶠᶜⁿ = new_data(FT, arch, (Face,   Center, Nothing), args...)
    σᶜᶠⁿ = new_data(FT, arch, (Center, Face,   Nothing), args...)
    σᶠᶠⁿ = new_data(FT, arch, (Face,   Face,   Nothing), args...)
    ηⁿ   = new_data(FT, arch, (Center, Center, Nothing), args...)
    ∂t_σ = new_data(FT, arch, (Center, Center, Nothing), args...)

    # Initialize: σ = 1 means z == r (identity mapping)
    for σ in (σᶜᶜ⁻, σᶜᶜⁿ, σᶠᶜⁿ, σᶜᶠⁿ, σᶠᶠⁿ)
        fill!(σ, 1)
    end

    return LR, ZStarVerticalCoordinate(rᵃᵃᶠ, rᵃᵃᶜ, Δrᵃᵃᶠ, Δrᵃᵃᶜ, ηⁿ, σᶜᶜⁿ, σᶠᶜⁿ, σᶜᶠⁿ, σᶠᶠⁿ, σᶜᶜ⁻, ∂t_σ)
end


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

@inline rnodes(grid::AUG, ℓz::F; with_halos=false, indices=Colon()) = view(_property(grid.z.cᵃᵃᶠ, ℓz, topology(grid, 3), grid.Nz, grid.Hz, with_halos), indices)
@inline rnodes(grid::AUG, ℓz::C; with_halos=false, indices=Colon()) = view(_property(grid.z.cᵃᵃᶜ, ℓz, topology(grid, 3), grid.Nz, grid.Hz, with_halos), indices)
@inline rnodes(grid::AUG, ℓx, ℓy, ℓz; with_halos=false, indices=Colon()) = rnodes(grid, ℓz; with_halos, indices)

@inline rnodes(grid::AUG, ::Nothing; kwargs...) = 1:1
@inline znodes(grid::AUG, ::Nothing; kwargs...) = 1:1

ZFlatAUG = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, Flat}
@inline rnodes(grid::ZFlatAUG, ℓz::F; with_halos=false, indices=Colon()) = _property(grid.z.cᵃᵃᶠ, ℓz, topology(grid, 3), grid.Nz, grid.Hz, with_halos)
@inline rnodes(grid::ZFlatAUG, ℓz::C; with_halos=false, indices=Colon()) = _property(grid.z.cᵃᵃᶜ, ℓz, topology(grid, 3), grid.Nz, grid.Hz, with_halos)

# TODO: extend in the Operators module
"""
    znodes(grid, ℓx, ℓy, ℓz, with_halos=false)

Return the positions over the interior nodes on `grid` in the ``z``-direction for the location `ℓx`,
`ℓy`, `ℓz`. For `Bounded` directions, `Face` nodes include the boundary points.

```jldoctest znodes
julia> using Oceananigans

julia> horz_periodic_grid = RectilinearGrid(size=(3, 3, 3), extent=(2π, 2π, 1), halo=(1, 1, 1),
                                            topology=(Periodic, Periodic, Bounded));

julia> z = znodes(horz_periodic_grid, Center())
-0.8333333333333334:0.3333333333333333:-0.16666666666666666

julia> z = znodes(horz_periodic_grid, Center(), Center(), Center())
-0.8333333333333334:0.3333333333333333:-0.16666666666666666

julia> z = znodes(horz_periodic_grid, Center(), Center(), Center(), with_halos=true)
5-element view(OffsetArray(::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, 0:4), :) with eltype Float64 with indices 0:4:
 -1.1666666666666667
 -0.8333333333333334
 -0.5
 -0.16666666666666666
  0.16666666666666666
```
"""
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
@inline cpu_face_constructor_z(grid::AbstractGeneralizedVerticalGrid) = GeneralizedVerticalDiscretization(cpu_face_constructor_r(grid))

####
#### Utilities
####

function validate_dimension_specification(T, ξ::GeneralizedVerticalDiscretization, dir, N, FT)
    r_faces_validated = validate_dimension_specification(T, ξ.r_faces, dir, N, FT)
    return GeneralizedVerticalDiscretization(r_faces_validated)
end
