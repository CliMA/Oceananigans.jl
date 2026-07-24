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

$(FIELDS)
"""
struct StaticVerticalDiscretization{C, D, E, F} <: AbstractVerticalCoordinate
    "Face-centered coordinate"
    cᵃᵃᶠ :: C
    "Cell-centered coordinate"
    cᵃᵃᶜ :: D
    "Face-centered grid spacing"
    Δᵃᵃᶠ :: E
    "Cell-centered grid spacing"
    Δᵃᵃᶜ :: F
end

# Summaries
const RegularStaticVerticalDiscretization  = StaticVerticalDiscretization{<:Any, <:Any, <:Number}
const AbstractStaticGrid  = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Any, <:StaticVerticalDiscretization}

coordinate_summary(topo, z::StaticVerticalDiscretization, name) = coordinate_summary(topo, z.Δᵃᵃᶜ, name)

"""
    struct MutableVerticalDiscretization{C, D, E, F, H, CC, FC, CF, FF} <: AbstractVerticalCoordinate

Represent a mutable vertical coordinate that can evolve in time.

Fields
======

$(FIELDS)
"""
struct MutableVerticalDiscretization{C, D, E, F, H, CC, FC, CF, FF} <: AbstractVerticalCoordinate
    "Face-centered reference coordinate"
    cᵃᵃᶠ :: C
    "Cell-centered reference coordinate"
    cᵃᵃᶜ :: D
    "Face-centered grid spacing"
    Δᵃᵃᶠ :: E
    "Cell-centered grid spacing"
    Δᵃᵃᶜ :: F
    "Surface elevation at the current time step"
    ηⁿ :: H
    "(Center, Center) scaling factor at the current time step"
    σᶜᶜⁿ :: CC
    "(Face, Center) scaling at the current time step"
    σᶠᶜⁿ :: FC
    "(Center, Face) scaling at the current time step"
    σᶜᶠⁿ :: CF
    "(Face, Face) scaling factor at the current time step"
    σᶠᶠⁿ :: FF
    "(Center, Center) scaling factor at the previous time step"
    σᶜᶜ⁻ :: CC
    "Time derivative of the cell-centered scaling factor"
    ∂t_σ :: CC
end

####
#### Some useful aliases
####

const RegularMutableVerticalDiscretization = MutableVerticalDiscretization{<:Any, <:Any, <:Number}
const RegularVerticalCoordinate = Union{RegularStaticVerticalDiscretization, RegularMutableVerticalDiscretization}

const AbstractMutableGrid = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Bounded, <:MutableVerticalDiscretization}
const RegularVerticalGrid = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Any,     <:RegularVerticalCoordinate}

is_static_discretization(::AbstractVerticalCoordinate) = true
is_static_discretization(::StaticVerticalDiscretization) = true
is_static_discretization(::MutableVerticalDiscretization) = false

"""
$(TYPEDSIGNATURES)

Construct a `MutableVerticalDiscretization` from `r_faces` that can be a `Tuple`,
a function of an index `k`, or an `AbstractArray`. A `MutableVerticalDiscretization`
defines a vertical coordinate that can evolve in time following certain rules.
Examples of `MutableVerticalDiscretization`s are the free-surface following coordinates
(also known as "zee-star") or the terrain following coordinates (also known as "sigma"
coordinates).
"""
MutableVerticalDiscretization(r_faces) =
    MutableVerticalDiscretization(r_faces, r_faces, (nothing for i in 1:9)...)

coordinate_summary(::Bounded, z::RegularMutableVerticalDiscretization, name) =
    @sprintf("regularly spaced with mutable Δr=%s", prettysummary(z.Δᵃᵃᶜ))

coordinate_summary(::Bounded, z::MutableVerticalDiscretization, name) =
    @sprintf("variably and mutably spaced with min(Δr)=%s, max(Δr)=%s",
             prettysummary(minimum(parent(z.Δᵃᵃᶜ))),
             prettysummary(maximum(parent(z.Δᵃᵃᶜ))))

function Base.show(io::IO, z::MutableVerticalDiscretization)
    print(io, "MutableVerticalDiscretization with reference interfaces r:\n")
    Base.show(io, z.cᵃᵃᶠ)
end

#####
##### Coordinate generation for grid constructors
#####

generate_coordinate(FT, ::Periodic, N, H, ::MutableVerticalDiscretization, coordinate_name, arch, args...) =
    throw(ArgumentError("Periodic domains are not supported for MutableVerticalDiscretization"))

# Generate a vertical coordinate with a scaling (`σ`) with respect to a reference coordinate `r` with spacing `Δr`.
# The grid might move with time, so the coordinate includes the time-derivative of the scaling `∂t_σ`.
# The value of the vertical coordinate at `Nz+1` is saved in `ηⁿ`.
function generate_coordinate(FT, topo, size, halo, coordinate::MutableVerticalDiscretization, coordinate_name, dim::Int, arch)

    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo

    if dim != 3
        msg = "MutableVerticalDiscretization is supported only in the third dimension (z)"
        throw(ArgumentError(msg))
    end

    if coordinate_name != :z
        msg = "MutableVerticalDiscretization is supported only for the z-coordinate"
        throw(ArgumentError(msg))
    end

    r_faces = coordinate.cᵃᵃᶠ

    LR, rᵃᵃᶠ, rᵃᵃᶜ, Δrᵃᵃᶠ, Δrᵃᵃᶜ = generate_coordinate(FT, topo[3](), Nz, Hz, r_faces, :r, arch)

    args = (topo, (Nx, Ny, Nz), (Hx, Hy, Hz))

    σᶜᶜ⁻ = new_data(FT, arch, (Center, Center, Nothing), args...)
    σᶜᶜⁿ = new_data(FT, arch, (Center, Center, Nothing), args...)
    σᶠᶜⁿ = new_data(FT, arch, (Face,   Center, Nothing), args...)
    σᶜᶠⁿ = new_data(FT, arch, (Center, Face,   Nothing), args...)
    σᶠᶠⁿ = new_data(FT, arch, (Face,   Face,   Nothing), args...)
    ηⁿ   = new_data(FT, arch, (Center, Center, Nothing), args...)
    ∂t_σ = new_data(FT, arch, (Center, Center, Nothing), args...)

    # Fill all the scalings with one for now (i.e. z == r)
    for σ in (σᶜᶜ⁻, σᶜᶜⁿ, σᶠᶜⁿ, σᶜᶠⁿ, σᶠᶠⁿ)
        fill!(σ, 1)
    end

    return LR, MutableVerticalDiscretization(rᵃᵃᶠ, rᵃᵃᶜ, Δrᵃᵃᶠ, Δrᵃᵃᶜ, ηⁿ, σᶜᶜⁿ, σᶠᶜⁿ, σᶜᶠⁿ, σᶠᶠⁿ, σᶜᶜ⁻, ∂t_σ)
end


####
#### Adapt and on_architecture
####

Adapt.adapt_structure(to, coord::StaticVerticalDiscretization) =
    StaticVerticalDiscretization(Adapt.adapt(to, coord.cᵃᵃᶠ),
                                 Adapt.adapt(to, coord.cᵃᵃᶜ),
                                 Adapt.adapt(to, coord.Δᵃᵃᶠ),
                                 Adapt.adapt(to, coord.Δᵃᵃᶜ))

Architectures.on_architecture(arch, coord::StaticVerticalDiscretization) =
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

Architectures.on_architecture(arch, coord::MutableVerticalDiscretization) =
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

"""
    zspacings(grid, ℓx, ℓy, ℓz)

Return a `KernelFunctionOperation` that computes the grid spacings for `grid`
in the ``z`` direction at location `ℓx, ℓy, ℓz`.

Examples
========
```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 4, 8), extent=(1, 1, 1));

julia> zspacings(grid, Center(), Center(), Face())
KernelFunctionOperation at (Center, Center, Face)
├── grid: 2×4×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── kernel_function: Δz (generic function with 19 methods)
└── arguments: ("Center", "Center", "Face")
```
"""
function zspacings end

"""
    rspacings(grid, ℓx, ℓy, ℓz)

Return a `KernelFunctionOperation` that computes the grid spacings for `grid`
in the ``r`` direction at location `ℓx, ℓy, ℓz`.

Examples
========
```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 4, 8), extent=(1, 1, 1));

julia> rspacings(grid, Center(), Center(), Face())
KernelFunctionOperation at (Center, Center, Face)
├── grid: 2×4×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── kernel_function: Δr (generic function with 19 methods)
└── arguments: ("Center", "Center", "Face")
```
"""
function rspacings end

# The 3-argument implementations of zspacings and rspacings are defined in
# src/AbstractOperations/grid_metrics.jl, where KernelFunctionOperation is available.
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
