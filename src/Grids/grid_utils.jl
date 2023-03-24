using CUDA
using Printf
using Base.Ryu: writeshortest
using LinearAlgebra: dot, cross
using OffsetArrays: IdOffsetRange

# Define default indices
default_indices(n) = Tuple(Colon() for i=1:n)

const BoundedTopology = Union{Bounded, LeftConnected}

"""
    topology(grid)

Return a tuple with the topology of the `grid` for each dimension.
"""
@inline topology(::AbstractGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = (TX, TY, TZ)

"""
    topology(grid, dim)

Return the topology of the `grid` for the `dim`-th dimension.
"""
@inline topology(grid, dim) = topology(grid)[dim]

"""
    architecture(grid::AbstractGrid)

Return the architecture (CPU or GPU) that the `grid` lives on.
"""
@inline architecture(grid::AbstractGrid) = grid.architecture

Base.eltype(::AbstractGrid{FT}) where FT = FT

function Base.:(==)(grid1::AbstractGrid, grid2::AbstractGrid)
    #check if grids are of the same type
    !isa(grid2, typeof(grid1).name.wrapper) && return false

    topology(grid1) !== topology(grid2) && return false

    x1, y1, z1 = nodes(grid1, (Face(), Face(), Face()))
    x2, y2, z2 = nodes(grid2, (Face(), Face(), Face()))

    return x1 == x2 && y1 == y2 && z1 == z2
end

const AT = AbstractTopology

Base.length(::Face,    ::BoundedTopology, N) = N + 1
Base.length(::Nothing, ::AT,              N) = 1
Base.length(::Face,    ::AT,              N) = N
Base.length(::Center,  ::AT,              N) = N
Base.length(::Nothing, ::Flat,            N) = N
Base.length(::Face,    ::Flat,            N) = N
Base.length(::Center,  ::Flat,            N) = N

# "Indices-aware" length
Base.length(loc, topo::AT, N, ::Colon) = length(loc, topo, N)
Base.length(loc, topo::AT, N, ind::UnitRange) = min(length(loc, topo, N), length(ind))

"""
    size(grid)

Return a 3-tuple of the number of "center" cells on a grid in (x, y, z).
Center cells have the location (Center, Center, Center).
"""
Base.size(grid::AbstractGrid) = (grid.Nx, grid.Ny, grid.Nz)

"""
    halo_size(grid)

Return a 3-tuple with the number of halo cells on either side of the
domain in (x, y, z).
"""
halo_size(grid) = (grid.Hx, grid.Hy, grid.Hz)
halo_size(grid, d) = halo_size(grid)[d]

Base.size(grid::AbstractGrid, d::Int) = size(grid)[d]

Base.size(grid::AbstractGrid, loc::Tuple, indices=default_indices(length(loc))) =
    size(loc, topology(grid), size(grid), indices)

function Base.size(loc, topo, sz, indices=default_indices(length(loc)))
    D = length(loc)
    return Tuple(length(instantiate(loc[d]), instantiate(topo[d]), sz[d], indices[d]) for d = 1:D)
end

Base.size(grid::AbstractGrid, loc::Tuple, d::Int) = size(grid, loc)[d]

"""
    total_length(loc, topo, N, H=0, ind=Colon())

Return the total length of a field at `loc`ation along
one dimension of `topo`logy with `N` centered cells and
`H` halo cells. If `ind` is provided the total_length
is restricted by `length(ind)`.
"""
total_length(::Face,    ::AT,              N, H=0) = N + 2H
total_length(::Center,  ::AT,              N, H=0) = N + 2H
total_length(::Face,    ::BoundedTopology, N, H=0) = N + 1 + 2H
total_length(::Nothing, ::AT,              N, H=0) = 1
total_length(::Nothing, ::Flat,            N, H=0) = N
total_length(::Face,    ::Flat,            N, H=0) = N
total_length(::Center,  ::Flat,            N, H=0) = N

# "Indices-aware" total length
total_length(loc, topo, N, H, ::Colon) = total_length(loc, topo, N, H)
total_length(loc, topo, N, H, ind::UnitRange) = min(total_length(loc, topo, N, H), length(ind))

total_size(a) = size(a) # fallback

"""
    total_size(grid, loc)

Return the "total" size of a `grid` at `loc`. This is a 3-tuple of integers
corresponding to the number of grid points along `x, y, z`.
"""
function total_size(loc, topo, sz, halo_sz, indices=default_indices(length(loc)))
    D = length(loc)
    return Tuple(total_length(instantiate(loc[d]), instantiate(topo[d]), sz[d], halo_sz[d], indices[d]) for d = 1:D)
end

total_size(grid::AbstractGrid, loc, indices=default_indices(length(loc))) =
    total_size(loc, topology(grid), size(grid), halo_size(grid), indices)

"""
    total_extent(topology, H, Δ, L)

Return the total extent, including halo regions, of constant-spaced
`Periodic` and `Flat` dimensions with number of halo points `H`,
constant grid spacing `Δ`, and interior extent `L`.
"""
@inline total_extent(topo, H, Δ, L) = L + (2H - 1) * Δ
@inline total_extent(::BoundedTopology, H, Δ, L) = L + 2H * Δ

# Grid domains
@inline domain(topo, N, ξ) = CUDA.@allowscalar ξ[1], ξ[N+1]
@inline domain(::Flat, N, ξ) = CUDA.@allowscalar ξ[1], ξ[1]

@inline x_domain(grid) = domain(topology(grid, 1)(), grid.Nx, grid.xᶠᵃᵃ)
@inline y_domain(grid) = domain(topology(grid, 2)(), grid.Ny, grid.yᵃᶠᵃ)
@inline z_domain(grid) = domain(topology(grid, 3)(), grid.Nz, grid.zᵃᵃᶠ)

regular_dimensions(grid) = ()

#####
##### << Indexing >>
#####

@inline left_halo_indices(loc, ::AT, N, H) = 1-H:0
@inline left_halo_indices(::Nothing, ::AT, N, H) = 1:0 # empty

@inline right_halo_indices(loc, ::AT, N, H) = N+1:N+H
@inline right_halo_indices(::Face, ::BoundedTopology, N, H) = N+2:N+1+H
@inline right_halo_indices(::Nothing, ::AT, N, H) = 1:0 # empty

@inline underlying_left_halo_indices(loc, ::AT, N, H) = 1:H
@inline underlying_left_halo_indices(::Nothing, ::AT, N, H) = 1:0 # empty

@inline underlying_right_halo_indices(loc,       ::AT, N, H) = N+1+H:N+2H
@inline underlying_right_halo_indices(::Face,    ::BoundedTopology, N, H) = N+2+H:N+1+2H
@inline underlying_right_halo_indices(::Nothing, ::AT, N, H) = 1:0 # empty

@inline interior_indices(loc,       ::AT,              N) = 1:N
@inline interior_indices(::Face,    ::BoundedTopology, N) = 1:N+1
@inline interior_indices(::Nothing, ::AT,              N) = 1:1

@inline interior_indices(::Nothing, ::Flat, N) = 1:N
@inline interior_indices(::Face,    ::Flat, N) = 1:N
@inline interior_indices(::Center,  ::Flat, N) = 1:N

@inline interior_x_indices(grid, loc) = interior_indices(loc[1], topology(grid, 1)(), size(grid, 1))
@inline interior_y_indices(grid, loc) = interior_indices(loc[2], topology(grid, 2)(), size(grid, 2))
@inline interior_z_indices(grid, loc) = interior_indices(loc[3], topology(grid, 3)(), size(grid, 3))

@inline interior_parent_offset(loc,       ::AT, H) = H
@inline interior_parent_offset(::Nothing, ::AT, H) = 0

@inline interior_parent_indices(::Nothing, ::AT,              N, H) = 1:1
@inline interior_parent_indices(::Face,    ::BoundedTopology, N, H) = 1+H:N+1+H
@inline interior_parent_indices(loc,       ::AT,              N, H) = 1+H:N+H

@inline interior_parent_indices(::Nothing, ::Flat, N, H) = 1:N
@inline interior_parent_indices(::Face,    ::Flat, N, H) = 1:N
@inline interior_parent_indices(::Center,  ::Flat, N, H) = 1:N

# All indices including halos.
@inline all_indices(::Nothing, ::AT,              N, H) = 1:1
@inline all_indices(::Face,    ::BoundedTopology, N, H) = 1-H:N+1+H
@inline all_indices(loc,       ::AT,              N, H) = 1-H:N+H

@inline all_indices(::Nothing, ::Flat, N, H) = 1:N
@inline all_indices(::Face,    ::Flat, N, H) = 1:N
@inline all_indices(::Center,  ::Flat, N, H) = 1:N

@inline all_x_indices(grid, loc) = all_indices(loc[1](), topology(grid, 1)(), size(grid, 1), halo_size(grid, 1))
@inline all_y_indices(grid, loc) = all_indices(loc[2](), topology(grid, 2)(), size(grid, 2), halo_size(grid, 2))
@inline all_z_indices(grid, loc) = all_indices(loc[3](), topology(grid, 3)(), size(grid, 3), halo_size(grid, 3))

@inline all_parent_indices(loc,       ::AT,              N, H) = 1:N+2H
@inline all_parent_indices(::Face,    ::BoundedTopology, N, H) = 1:N+1+2H
@inline all_parent_indices(::Nothing, ::AT,              N, H) = 1:1

@inline all_parent_indices(::Nothing, ::Flat, N, H) = 1:N
@inline all_parent_indices(::Face,    ::Flat, N, H) = 1:N
@inline all_parent_indices(::Center,  ::Flat, N, H) = 1:N

@inline all_parent_x_indices(grid, loc) = all_parent_indices(loc[1](), topology(grid, 1)(), size(grid, 1), halo_size(grid, 1))
@inline all_parent_y_indices(grid, loc) = all_parent_indices(loc[2](), topology(grid, 2)(), size(grid, 2), halo_size(grid, 2))
@inline all_parent_z_indices(grid, loc) = all_parent_indices(loc[3](), topology(grid, 3)(), size(grid, 3), halo_size(grid, 3))

parent_index_range(::Colon,                       loc, topo, halo) = Colon()
parent_index_range(::Base.Slice{<:IdOffsetRange}, loc, topo, halo) = Colon()
parent_index_range(index::UnitRange,              loc, topo, halo) = index .+ interior_parent_offset(loc, topo, halo)

parent_index_range(index::UnitRange, ::Nothing, ::Flat, halo) = index
parent_index_range(index::UnitRange, ::Nothing, ::AT,   halo) = 1:1 # or Colon()

index_range_offset(index::UnitRange, loc, topo, halo) = index[1] - interior_parent_offset(loc, topo, halo)
index_range_offset(::Colon, loc, topo, halo)          = - interior_parent_offset(loc, topo, halo)

@inline cpu_face_constructor_x(grid) = Array(xnodes(grid, Face(); with_halos=true)[1:size(grid, 1)+1])
@inline cpu_face_constructor_y(grid) = Array(ynodes(grid, Face(); with_halos=true)[1:size(grid, 2)+1])
@inline cpu_face_constructor_z(grid) = Array(znodes(grid, Face(); with_halos=true)[1:size(grid, 3)+1])

#####
##### << Nodes >>
#####

@inline node(i, j, k, grid, ℓx, ℓy, ℓz) = (xnode(i, j, k, grid, ℓx, ℓy, ℓz),
                                           ynode(i, j, k, grid, ℓx, ℓy, ℓz),
                                           znode(i, j, k, grid, ℓx, ℓy, ℓz))

@inline node(i, j, k, grid, ℓx::Nothing, ℓy, ℓz) = (ynode(i, j, k, grid, ℓx, ℓy, ℓz), znode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline node(i, j, k, grid, ℓx, ℓy::Nothing, ℓz) = (xnode(i, j, k, grid, ℓx, ℓy, ℓz), znode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline node(i, j, k, grid, ℓx, ℓy, ℓz::Nothing) = (xnode(i, j, k, grid, ℓx, ℓy, ℓz), ynode(i, j, k, grid, ℓx, ℓy, ℓz))

@inline node(i, j, k, grid, ℓx, ℓy::Nothing, ℓz::Nothing) = tuple(xnode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline node(i, j, k, grid, ℓx::Nothing, ℓy, ℓz::Nothing) = tuple(ynode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline node(i, j, k, grid, ℓx::Nothing, ℓy::Nothing, ℓz) = tuple(znode(i, j, k, grid, ℓx, ℓy, ℓz))

xnodes(grid, ::Nothing; kwargs...) = 1:1
ynodes(grid, ::Nothing; kwargs...) = 1:1
znodes(grid, ::Nothing; kwargs...) = 1:1

"""
    xnodes(grid, ℓx, ℓy, ℓz, with_halos=false)

Return the positions over the interior nodes on `grid` in the ``x``-direction for the location `ℓx`,
`ℓy`, `ℓz`. For `Bounded` directions, `Face` nodes include the boundary points.

See [`znodes`](@ref) for examples.
"""
@inline xnodes(grid, ℓx, ℓy, ℓz; kwargs...) = xnodes(grid, ℓx; kwargs...)

"""
    ynodes(grid, ℓx, ℓy, ℓz, with_halos=false)

Return the positions over the interior nodes on `grid` in the ``y``-direction for the location `ℓx`,
`ℓy`, `ℓz`. For `Bounded` directions, `Face` nodes include the boundary points.

See [`znodes`](@ref) for examples.
"""
@inline ynodes(grid, ℓx, ℓy, ℓz; kwargs...) = ynodes(grid, ℓy; kwargs...)

"""
    znodes(grid, ℓx, ℓy, ℓz; with_halos=false)

Return the positions over the interior nodes on `grid` in the ``z``-direction for the location `ℓx`,
`ℓy`, `ℓz`. For `Bounded` directions, `Face` nodes include the boundary points.

```jldoctest znodes
julia> using Oceananigans

julia> horz_periodic_grid = RectilinearGrid(size=(3, 3, 3), extent=(2π, 2π, 1), halo=(1, 1, 1),
                                            topology=(Periodic, Periodic, Bounded));

julia> zC = znodes(horz_periodic_grid, Center())
3-element view(OffsetArray(::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, 0:4), 1:3) with eltype Float64:
 -0.8333333333333334
 -0.5
 -0.16666666666666666

julia> zC = znodes(horz_periodic_grid, Center(), Center(), Center())
3-element view(OffsetArray(::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, 0:4), 1:3) with eltype Float64:
 -0.8333333333333334
 -0.5
 -0.16666666666666666

julia> zC = znodes(horz_periodic_grid, Center(), Center(), Center(), with_halos=true)
-1.1666666666666667:0.3333333333333333:0.16666666666666666 with indices 0:4
```
"""
@inline znodes(grid, ℓx, ℓy, ℓz; kwargs...) = znodes(grid, ℓz; kwargs...)

"""
    nodes(grid, (ℓx, ℓy, ℓz); reshape=false, with_halos=false)
    nodes(grid, ℓx, ℓy, ℓz; reshape=false, with_halos=false)

Return a 3-tuple of views over the interior nodes
at the locations in `loc=(ℓx, ℓy, ℓz)` in `x, y, z`.

If `reshape=true`, the views are reshaped to 3D arrays
with non-singleton dimensions 1, 2, 3 for `x, y, z`, respectively.
These reshaped arrays can then be used in broadcast operations with 3D fields
or arrays.

See [`xnodes`](@ref), [`ynodes`](@ref), and [`znodes`](@ref).
"""
function nodes(grid::AbstractGrid, ℓx, ℓy, ℓz; reshape=false, with_halos=false)
    x = xnodes(grid, ℓx, ℓy, ℓz; with_halos)
    y = ynodes(grid, ℓx, ℓy, ℓz; with_halos)
    z = znodes(grid, ℓx, ℓy, ℓz; with_halos)

    if reshape
        N = (length(x), length(y), length(z))
        x = Base.reshape(x, N[1], 1, 1)
        y = Base.reshape(y, 1, N[2], 1)
        z = Base.reshape(z, 1, 1, N[3])
    end

    return (x, y, z)
end

nodes(grid::AbstractGrid, (ℓx, ℓy, ℓz); reshape=false, with_halos=false) = nodes(grid, ℓx, ℓy, ℓz; reshape, with_halos)


#####
##### << Spacings >>
#####

# placeholders; see Oceananigans.Operators for x/y/zspacing definitions
function xspacing end
function yspacing end
function zspacing end

"""
    xspacings(grid, ℓx, ℓy, ℓz; with_halos=true)

Return the spacings over the interior nodes on `grid` in the ``x``-direction for the location `ℓx`,
`ℓy`, `ℓz`. For `Bounded` directions, `Face` nodes include the boundary points.

```jldoctest xspacings
julia> using Oceananigans

julia> grid = LatitudeLongitudeGrid(size=(8, 15, 10), longitude=(-20, 60), latitude=(-10, 50), z=(-100, 0));

julia> xspacings(grid, Center(), Face(), Center())
16-element view(OffsetArray(::Vector{Float64}, -2:18), 1:16) with eltype Float64:
      1.0950562585518518e6
      1.1058578920188267e6
      1.1112718969963323e6
      1.1112718969963323e6
      1.1058578920188267e6
      1.0950562585518518e6
      1.0789196210678827e6
      ⋮
 999413.38046802
 962976.3124613502
 921847.720658409
 876227.979424229
 826339.3435524226
 772424.8654621692
 714747.2110712599
 ```
"""
@inline xspacings(grid, ℓx, ℓy, ℓz; with_halos=true) = xspacings(grid, ℓx; with_halos)


"""
    yspacings(grid, ℓx, ℓy, ℓz; with_halos=true)

Return the spacings over the interior nodes on `grid` in the ``y``-direction for the location `ℓx`,
`ℓy`, `ℓz`. For `Bounded` directions, `Face` nodes include the boundary points.

```jldoctest yspacings
julia> using Oceananigans

julia> grid = LatitudeLongitudeGrid(size=(20, 15, 10), longitude=(0, 20), latitude=(-15, 15), z=(-100, 0));

julia> yspacings(grid, Center(), Center(), Center())
222389.85328911748
```
"""
@inline yspacings(grid, ℓx, ℓy, ℓz; with_halos=true) = yspacings(grid, ℓy; with_halos)

"""
    zspacings(grid, ℓx, ℓy, ℓz; with_halos=true)

Return the spacings over the interior nodes on `grid` in the ``z``-direction for the location `ℓx`,
`ℓy`, `ℓz`. For `Bounded` directions, `Face` nodes include the boundary points.

```jldoctest zspacings
julia> using Oceananigans

julia> grid = LatitudeLongitudeGrid(size=(20, 15, 10), longitude=(0, 20), latitude=(-15, 15), z=(-100, 0));

julia> zspacings(grid, Center(), Center(), Center())
10.0
```
"""
@inline zspacings(grid, ℓx, ℓy, ℓz; with_halos=true) = zspacings(grid, ℓz; with_halos)

destantiate(::Face)   = Face
destantiate(::Center) = Center

function minimum_spacing(dir, grid, ℓx, ℓy, ℓz)
    spacing = eval(Symbol(dir, :spacing))
    LX, LY, LZ = map(destantiate, (ℓx, ℓy, ℓz))
    Δ = KernelFunctionOperation{LX, LY, LZ}(spacing, grid, ℓx, ℓy, ℓz)

    return minimum(Δ)
end

"""
    minimum_xspacing(grid, ℓx, ℓy, ℓz)
    minimum_xspacing(grid) = minimum_xspacing(grid, Center(), Center(), Center())

Return the minimum spacing for `grid` in ``x`` direction at location `ℓx, ℓy, ℓz`.

Examples
========
```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 4, 8), extent=(1, 1, 1));

julia> minimum_xspacing(grid, Center(), Center(), Center())
0.5
```
"""
minimum_xspacing(grid, ℓx, ℓy, ℓz) = minimum_spacing(:x, grid, ℓx, ℓy, ℓz)
minimum_xspacing(grid) = minimum_spacing(:x, grid, Center(), Center(), Center())
"""
    minimum_yspacing(grid, ℓx, ℓy, ℓz)
    minimum_yspacing(grid) = minimum_yspacing(grid, Center(), Center(), Center())

Return the minimum spacing for `grid` in ``y`` direction at location `ℓx, ℓy, ℓz`.

Examples
========
```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 4, 8), extent=(1, 1, 1));

julia> minimum_yspacing(grid, Center(), Center(), Center())
0.25
```
"""
minimum_yspacing(grid, ℓx, ℓy, ℓz) = minimum_spacing(:y, grid, ℓx, ℓy, ℓz)
minimum_yspacing(grid) = minimum_spacing(:y, grid, Center(), Center(), Center())

"""
    minimum_zspacing(grid, ℓx, ℓy, ℓz)
    minimum_zspacing(grid) = minimum_zspacing(grid, Center(), Center(), Center())

Return the minimum spacing for `grid` in ``z`` direction at location `ℓx, ℓy, ℓz`.

Examples
========
```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 4, 8), extent=(1, 1, 1));

julia> minimum_zspacing(grid, Center(), Center(), Center())
0.125
```
"""
minimum_zspacing(grid, ℓx, ℓy, ℓz) = minimum_spacing(:z, grid, ℓx, ℓy, ℓz)
minimum_zspacing(grid) = minimum_spacing(:z, grid, Center(), Center(), Center())

#####
##### Convenience functions
#####

unpack_grid(grid) = grid.Nx, grid.Ny, grid.Nz, grid.Lx, grid.Ly, grid.Lz

flatten_halo(TX, TY, TZ, halo) = Tuple(T === Flat ? 0 : halo[i] for (i, T) in enumerate((TX, TY, TZ)))
flatten_size(TX, TY, TZ, halo) = Tuple(T === Flat ? 0 : halo[i] for (i, T) in enumerate((TX, TY, TZ)))

"""
    pop_flat_elements(tup, topo)

Return a new tuple that contains the elements of `tup`,
except for those elements corresponding to the `Flat` directions
in `topo`.
"""
function pop_flat_elements(tup, topo)
    new_tup = []
    for i = 1:3
        topo[i] != Flat && push!(new_tup, tup[i])
    end
    return Tuple(new_tup)
end

#####
##### Directions (for tilted domains)
#####

struct ZDirection end

Base.summary(::ZDirection) = "ZDirection()"
Base.show(io::IO, zdir::ZDirection) = print(io, summary(zdir))

struct NegativeZDirection end

Base.summary(::NegativeZDirection) = "NegativeZDirection()"
Base.show(io::IO, zdir::NegativeZDirection) = print(io, summary(zdir))

-(::NegativeZDirection) = ZDirection()
-(::ZDirection) = NegativeZDirection()

#####
##### Show utils
#####

size_summary(sz) = string(sz[1], "×", sz[2], "×", sz[3])
prettysummary(σ::AbstractFloat, plus=false) = writeshortest(σ, plus, false, true, -1, UInt8('e'), false, UInt8('.'), false, true)
dimension_summary(topo::Flat, name, args...) = "Flat $name"

function domain_summary(topo, name, left, right)
    interval = (topo isa Bounded) ||
               (topo isa LeftConnected) ? "]" : ")"

    topo_string = topo isa Periodic ? "Periodic " :
                  topo isa Bounded ? "Bounded  " :
                  topo isa FullyConnected ? "FullyConnected " :
                  topo isa LeftConnected ? "LeftConnected  " :
                  "RightConnected "

    return string(topo_string, name, " ∈ [",
                  prettysummary(left), ", ",
                  prettysummary(right), interval)
end

function dimension_summary(topo, name, left, right, spacing, pad_domain=0)
    prefix = domain_summary(topo, name, left, right)
    padding = " "^(pad_domain+1) 
    return string(prefix, padding, coordinate_summary(spacing, name))
end

coordinate_summary(Δ::Number, name) = @sprintf("regularly spaced with Δ%s=%s", name, prettysummary(Δ))

coordinate_summary(Δ::Union{AbstractVector, AbstractMatrix}, name) =
    @sprintf("variably spaced with min(Δ%s)=%s, max(Δ%s)=%s",
             name, prettysummary(minimum(parent(Δ))),
             name, prettysummary(maximum(parent(Δ))))

#####
##### Spherical geometry
#####

"""
    spherical_area_triangle(a::Number, b::Number, c::Number)

Return the area of a spherical triangle on the unit sphere with sides `a`, `b`, and `c`.

The area of a spherical triangle on the unit sphere is ``E = A + B + C - π``, where ``A``, ``B``, and ``C``
are the triangle's inner angles.

It has been known since the time of Euler and Lagrange that
``\\tan(E/2) = P / (1 + \\cos a + \\cos b + \\cos c)``, where
``P = (1 - \\cos²a - \\cos²b - \\cos²c + 2 \\cos a \\cos b \\cos c)^{1/2}``.

References
==========
* Euler, L. (1778) De mensura angulorum solidorum, Opera omnia, 26, 204-233 (Orig. in Acta adac. sc. Petrop. 1778)
* Lagrange,  J.-L. (1798) Solutions de quilquies problèmes relatifs au triangles sphéruques, Oeuvres, 7, 331-359.
"""
function spherical_area_triangle(a::Number, b::Number, c::Number)
    cosa = cos(a)
    cosb = cos(b)
    cosc = cos(c)

    tan½E = sqrt(1 - cosa^2 - cosb^2 - cosc^2 + 2cosa * cosb * cosc)
    tan½E /= 1 + cosa + cosb + cosc

    return 2atan(tan½E)
end

"""
    spherical_area_triangle(a::AbstractVector, b::AbstractVector, c::AbstractVector)

Return the area of a spherical triangle on the unit sphere with vertices given by the 3-vectors
`a`, `b`, and `c` whose origin is the the center of the sphere. The formula was first given by
Eriksson (1990).

If we denote with ``A``, ``B``, and ``C`` the inner angles of the spherical triangle and with
``a``, ``b``, and ``c`` the side of the triangle then, it has been known since Euler and Lagrange
that ``\\tan(E/2) = P / (1 + \\cos a + \\cos b + \\cos c)``, where ``E = A + B + C - π`` is the
triangle's excess and ``P = (1 - \\cos²a - \\cos²b - \\cos²c + 2 \\cos a \\cos b \\cos c)^{1/2}``.
On the unit sphere, ``E`` is precisely the area of the spherical triangle. Erikkson (1990) showed
that ``P`` above is the same as the volume defined by the vectors `a`, `b`, and `c`, that is
``P = |𝐚 \\cdot (𝐛 \\times 𝐜)|``.

References
==========
* Eriksson, F. (1990) On the measure of solid angles, Mathematics Magazine, 63 (3), 184-187, doi:10.1080/0025570X.1990.11977515
"""
function spherical_area_triangle(a₁::AbstractVector, a₂::AbstractVector, a₃::AbstractVector)
    (sum(a₁.^2) ≈ 1 && sum(a₂.^2) ≈ 1 && sum(a₃.^2) ≈ 1) || error("a₁, a₂, a₃ must be unit vectors")

    tan½E = abs(dot(a₁, cross(a₂, a₃)))
    tan½E /= 1 + dot(a₁, a₂) + dot(a₂, a₃) + dot(a₁, a₃)

    return 2atan(tan½E)
end

"""
    spherical_area_quadrilateral(a₁, a₂, a₃, a₄)

Return the area of a spherical quadrilateral on the unit sphere whose points are given by 3-vectors,
`a`, `b`, `c`, and `d`. The area of the quadrilateral is given as the sum of the ares of the two
non-overlapping triangles. To avoid having to pick the triangles appropriately ensuring they are not
overlapping, we compute the area of the quadrilateral as the half the sum of the areas of all four potential
triangles formed by `a₁`, `a₂`, `a₃`, and `a₄`.
"""
spherical_area_quadrilateral(a::AbstractVector, b::AbstractVector, c::AbstractVector, d::AbstractVector) =
    1/2 * (spherical_area_triangle(a, b, c) + spherical_area_triangle(a, b, d) +
           spherical_area_triangle(a, c, d) + spherical_area_triangle(b, c, d))

"""
    add_halos(data, loc, topo, sz, halo_sz; warnings=true)

Add halos of size `halo_sz :: NTuple{3}{Int}` to `data` that corresponds to
size `sz :: NTuple{3}{Int}`, location `loc :: NTuple{3}`, and topology
`topo :: NTuple{3}`.

Setting the keyword `warning = false` will spare you from warnings regarding
the size of `data` being too big or too small for the `loc`, `topo`, and `sz`
provided.

Example
=======

```julia
julia> using Oceananigans

julia> using Oceananigans.Grids: add_halos, total_length

julia> Nx, Ny, Nz = (3, 3, 1);

julia> loc = (Face, Center, Nothing);

julia> topo = (Bounded, Periodic, Bounded);

julia> data = rand(total_length(loc[1](), topo[1](), Nx, 0), total_length(loc[2](), topo[2](), Ny, 0))
4×3 Matrix{Float64}:
 0.771924  0.998196   0.48775
 0.499878  0.470224   0.669928
 0.254603  0.73885    0.0821657
 0.997512  0.0440224  0.726334

julia> add_halos(data, loc, topo, (Nx, Ny, Nz), (1, 2, 0))
6×7 OffsetArray(::Matrix{Float64}, 0:5, -1:5) with eltype Float64 with indices 0:5×-1:5:
 0.0  0.0  0.0       0.0        0.0        0.0  0.0
 0.0  0.0  0.771924  0.998196   0.48775    0.0  0.0
 0.0  0.0  0.499878  0.470224   0.669928   0.0  0.0
 0.0  0.0  0.254603  0.73885    0.0821657  0.0  0.0
 0.0  0.0  0.997512  0.0440224  0.726334   0.0  0.0
 0.0  0.0  0.0       0.0        0.0        0.0  0.0

 julia> data = rand(8, 2)
8×2 Matrix{Float64}:
 0.910064  0.491983
 0.597547  0.775168
 0.711421  0.519057
 0.697258  0.450122
 0.300358  0.510102
 0.865862  0.579322
 0.196049  0.217199
 0.799729  0.822402

julia> add_halos(data, loc, topo, (Nx, Ny, Nz), (1, 2, 0))
┌ Warning: data has larger size than expected in first dimension; some data is lost
└ @ Oceananigans.Grids ~/Oceananigans.jl/src/Grids/grid_utils.jl:650
┌ Warning: data has smaller size than expected in second dimension; rest of entries are filled with zeros.
└ @ Oceananigans.Grids ~/Oceananigans.jl/src/Grids/grid_utils.jl:655
6×7 OffsetArray(::Matrix{Float64}, 0:5, -1:5) with eltype Float64 with indices 0:5×-1:5:
 0.0  0.0  0.0       0.0       0.0  0.0  0.0
 0.0  0.0  0.910064  0.491983  0.0  0.0  0.0
 0.0  0.0  0.597547  0.775168  0.0  0.0  0.0
 0.0  0.0  0.711421  0.519057  0.0  0.0  0.0
 0.0  0.0  0.697258  0.450122  0.0  0.0  0.0
 0.0  0.0  0.0       0.0       0.0  0.0  0.0
```
"""
function add_halos(data, loc, topo, sz, halo_sz; warnings=true)

    Nx, Ny, Nz = size(data)

    arch = architecture(data)

    # bring to CPU
    map(a -> arch_array(CPU(), a), data)

    nx, ny, nz = total_length(loc[1](), topo[1](), sz[1], 0),
                 total_length(loc[2](), topo[2](), sz[2], 0),
                 total_length(loc[3](), topo[3](), sz[3], 0)

    if warnings
        Nx > nx && @warn("data has larger size than expected in first dimension; some data is lost")
        Ny > ny && @warn("data has larger size than expected in second dimension; some data is lost")
        Nz > nz && @warn("data has larger size than expected in third dimension; some data is lost")

        Nx < nx && @warn("data has smaller size than expected in first dimension; rest of entries are filled with zeros.")
        Ny < ny && @warn("data has smaller size than expected in second dimension; rest of entries are filled with zeros.")
        Nz < nz && @warn("data has smaller size than expected in third dimension; rest of entries are filled with zeros.")
    end

    offset_array = dropdims(new_data(eltype(data), CPU(), loc, topo, sz, halo_sz), dims=3)

    nx = minimum((nx, Nx))
    ny = minimum((ny, Ny))
    nz = minimum((nz, Nz))

    offset_array[1:nx, 1:ny, 1:nz] = data[1:nx, 1:ny, 1:nz]

    # return to data's original architecture 
    map(a -> arch_array(arch, a), offset_array)

    return offset_array
end

function add_halos(data::AbstractArray{FT, 2} where FT, loc, topo, sz, halo_sz; warnings=true)
    Nx, Ny = size(data)
    return add_halos(reshape(data, (Nx, Ny, 1)), loc, topo, sz, halo_sz; warnings)
end
