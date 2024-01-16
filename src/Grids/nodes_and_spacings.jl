#####
##### node
#####
##### All grids should define the functions ξnode, ηnode, and rnode representing
##### the first, second, and third coordinates respectively.
#####

node_names(grid, ℓx, ℓy, ℓz) = _node_names(grid, ℓx, ℓy, ℓz)

node_names(grid::XFlatGrid, ℓx, ℓy, ℓz)   = _node_names(grid, nothing, ℓy, ℓz)
node_names(grid::YFlatGrid, ℓx, ℓy, ℓz)   = _node_names(grid, ℓx, nothing, ℓz)
node_names(grid::ZFlatGrid, ℓx, ℓy, ℓz)   = _node_names(grid, ℓx, ℓy, nothing)
node_names(grid::XYFlatGrid, ℓx, ℓy, ℓz)  = _node_names(grid, nothing, nothing, ℓz)
node_names(grid::XZFlatGrid, ℓx, ℓy, ℓz)  = _node_names(grid, nothing, ℓy, nothing)
node_names(grid::YZFlatGrid, ℓx, ℓy, ℓz)  = _node_names(grid, ℓx, nothing, nothing)
node_names(grid::XYZFlatGrid, ℓx, ℓy, ℓz) = _node_names(grid, nothing, nothing, nothing)

_node_names(grid, ℓx, ℓy, ℓz) = (ξname(grid), ηname(grid), rname(grid))

_node_names(grid, ::Nothing, ℓy, ℓz) = (ηname(grid), rname(grid))
_node_names(grid, ℓx, ::Nothing, ℓz) = (ξname(grid), rname(grid))
_node_names(grid, ℓx, ℓy, ::Nothing) = (ξname(grid), ηname(grid))

_node_names(grid, ℓx, ::Nothing, ::Nothing) = tuple(ξname(grid))
_node_names(grid, ::Nothing, ℓy, ::Nothing) = tuple(ηname(grid))
_node_names(grid, ::Nothing, ::Nothing, ℓz) = tuple(rname(grid))

_node_names(grid, ::Nothing, ::Nothing, ::Nothing) = tuple()

# Interface for grids to opt-in to `node`: ξnode, ηnode, rnode
@inline _node(i, j, k, grid, ℓx, ℓy, ℓz) = (ξnode(i, j, k, grid, ℓx, ℓy, ℓz),
                                            ηnode(i, j, k, grid, ℓx, ℓy, ℓz),
                                            rnode(i, j, k, grid, ℓx, ℓy, ℓz))

# Omission of Nothing locations
@inline _node(i, j, k, grid, ℓx::Nothing, ℓy, ℓz) = (ηnode(i, j, k, grid, ℓx, ℓy, ℓz), rnode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline _node(i, j, k, grid, ℓx, ℓy::Nothing, ℓz) = (ξnode(i, j, k, grid, ℓx, ℓy, ℓz), rnode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline _node(i, j, k, grid, ℓx, ℓy, ℓz::Nothing) = (ξnode(i, j, k, grid, ℓx, ℓy, ℓz), ηnode(i, j, k, grid, ℓx, ℓy, ℓz))

@inline _node(i, j, k, grid, ℓx, ℓy::Nothing, ℓz::Nothing) = tuple(ξnode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline _node(i, j, k, grid, ℓx::Nothing, ℓy, ℓz::Nothing) = tuple(ηnode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline _node(i, j, k, grid, ℓx::Nothing, ℓy::Nothing, ℓz) = tuple(rnode(i, j, k, grid, ℓx, ℓy, ℓz))

@inline _node(i, j, k, grid, ::Nothing, ::Nothing, ::Nothing) = tuple()

# Omission of Flat directions by "nullifying" locations in Flat directions
@inline node(i, j, k, grid, ℓx, ℓy, ℓz) = _node(i, j, k, grid, ℓx, ℓy, ℓz)

@inline node(i, j, k, grid::XFlatGrid, ℓx, ℓy, ℓz) = _node(i, j, k, grid, nothing, ℓy, ℓz)
@inline node(i, j, k, grid::YFlatGrid, ℓx, ℓy, ℓz) = _node(i, j, k, grid, ℓx, nothing, ℓz)
@inline node(i, j, k, grid::ZFlatGrid, ℓx, ℓy, ℓz) = _node(i, j, k, grid, ℓx, ℓy, nothing)

@inline node(i, j, k, grid::XYFlatGrid, ℓx, ℓy, ℓz) = _node(i, j, k, grid, nothing, nothing, ℓz)
@inline node(i, j, k, grid::XZFlatGrid, ℓx, ℓy, ℓz) = _node(i, j, k, grid, nothing, ℓy, nothing)
@inline node(i, j, k, grid::YZFlatGrid, ℓx, ℓy, ℓz) = _node(i, j, k, grid, ℓx, nothing, nothing)

@inline node(i, j, k, grid::XYZFlatGrid, ℓx, ℓy, ℓz) = tuple()

#####
##### << Nodes >>
#####

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
    λnodes(grid::AbstractCurvilinearGrid, ℓx, ℓy, ℓz, with_halos=false)

Return the positions over the interior nodes on a curvilinear `grid` in the ``λ``-direction
for the location `ℓλ`, `ℓφ`, `ℓz`. For `Bounded` directions, `Face` nodes include the boundary points.

See [`znodes`](@ref) for examples.
"""
@inline λnodes(grid::AbstractCurvilinearGrid, ℓλ, ℓφ, ℓz; kwargs...) = λnodes(grid, ℓλ; kwargs...)

"""
    φnodes(grid::AbstractCurvilinearGrid, ℓx, ℓy, ℓz, with_halos=false)

Return the positions over the interior nodes on a curvilinear `grid` in the ``φ``-direction
for the location `ℓλ`, `ℓφ`, `ℓz`. For `Bounded` directions, `Face` nodes include the boundary points.

See [`znodes`](@ref) for examples.
"""
@inline φnodes(grid::AbstractCurvilinearGrid, ℓλ, ℓφ, ℓz; kwargs...) = φnodes(grid, ℓφ; kwargs...)

"""
    nodes(grid, (ℓx, ℓy, ℓz); reshape=false, with_halos=false)
    nodes(grid, ℓx, ℓy, ℓz; reshape=false, with_halos=false)

Return a 3-tuple of views over the interior nodes of the `grid`'s
native coordinates at the locations in `loc=(ℓx, ℓy, ℓz)` in `x, y, z`.

If `reshape=true`, the views are reshaped to 3D arrays with non-singleton
dimensions 1, 2, 3 for `x, y, z`, respectively. These reshaped arrays can then
be used in broadcast operations with 3D fields or arrays.

For `RectilinearGrid`s the native coordinates are `x, y, z`; for curvilinear grids,
like `LatitudeLongitudeGrid` or `OrthogonalSphericalShellGrid` the native coordinates
are `λ, φ, z`.

See [`xnodes`](@ref), [`ynodes`](@ref), [`znodes`](@ref), [`λnodes`](@ref), and [`φnodes`](@ref).
"""
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
      1.0575265956426917e6
      1.0309814069457315e6
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
