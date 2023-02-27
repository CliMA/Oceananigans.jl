using CUDA
using Printf
using Base.Ryu: writeshortest
using LinearAlgebra: dot, cross
using OffsetArrays: IdOffsetRange

#####
##### Convenience functions
#####

const BoundedTopology = Union{Bounded, LeftConnected}

Base.length(::Type{Face}, topo, N) = N
Base.length(::Type{Face}, ::Type{<:BoundedTopology}, N) = N+1
Base.length(::Type{Center}, topo, N) = N
Base.length(::Type{Nothing}, topo, N) = 1

Base.length(::Type{Nothing}, ::Type{Flat}, N) = N
Base.length(::Type{Face},    ::Type{Flat}, N) = N
Base.length(::Type{Center},  ::Type{Flat}, N) = N

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

"""
    Constant Grid Definitions 
"""

Base.eltype(::AbstractGrid{FT}) where FT = FT

function Base.:(==)(grid1::AbstractGrid, grid2::AbstractGrid)
    #check if grids are of the same type
    !isa(grid2, typeof(grid1).name.wrapper) && return false

    topology(grid1) !== topology(grid2) && return false

    x1, y1, z1 = nodes((Face, Face, Face), grid1)
    x2, y2, z2 = nodes((Face, Face, Face), grid2)

    CUDA.@allowscalar return x1 == x2 && y1 == y2 && z1 == z2
end

"""
    size(loc, grid)

Return the size of a `grid` at `loc`, not including halos.
This is a 3-tuple of integers corresponding to the number of interior nodes
along `x, y, z`.
"""
Base.size(loc, grid::AbstractGrid) = (length(loc[1], topology(grid, 1), grid.Nx),
                                      length(loc[2], topology(grid, 2), grid.Ny),
                                      length(loc[3], topology(grid, 3), grid.Nz))

Base.size(grid::AbstractGrid) = size((Center, Center, Center), grid)
Base.size(grid::AbstractGrid, d) = size(grid)[d]
Base.size(loc, grid, d) = size(loc, grid)[d]

total_size(a) = size(a) # fallback

"""
    total_size(loc, grid)

Return the "total" size of a `grid` at `loc`. This is a 3-tuple of integers
corresponding to the number of grid points along `x, y, z`.
"""
total_size(loc, grid) = (total_length(loc[1], topology(grid, 1), grid.Nx, grid.Hx),
                         total_length(loc[2], topology(grid, 2), grid.Ny, grid.Hy),
                         total_length(loc[3], topology(grid, 3), grid.Nz, grid.Hz))


function total_size(loc, grid, indices::Tuple)
    sz = total_size(loc, grid)
    return Tuple(ind isa Colon ? sz[i] : min(length(ind), sz[i]) for (i, ind) in enumerate(indices))
end

function Base.size(loc, grid::AbstractGrid, indices::Tuple)
    sz = size(loc, grid)
    return Tuple(ind isa Colon ? sz[i] : min(length(ind), sz[i]) for (i, ind) in enumerate(indices))
end



"""
    halo_size(grid)

Return a tuple with the size of the halo in each dimension.
"""
halo_size(grid) = (grid.Hx, grid.Hy, grid.Hz)

"""
    total_extent(topology, H, Δ, L)

Return the total extent, including halo regions, of constant-spaced
`Periodic` and `Flat` dimensions with number of halo points `H`,
constant grid spacing `Δ`, and interior extent `L`.
"""
@inline total_extent(topology, H, Δ, L) = L + (2H - 1) * Δ
@inline total_extent(::Type{<:BoundedTopology}, H, Δ, L) = L + 2H * Δ

"""
    total_length(loc, topo, N, H=0)

Return the total length of a field at `loc`ation along
one dimension of `topo`logy with `N` centered cells and
`H` halo cells.
"""
@inline total_length(loc,             topo,            N, H=0) = N + 2H
@inline total_length(::Type{Face},    ::Type{<:BoundedTopology}, N, H=0) = N + 1 + 2H
@inline total_length(::Type{Nothing}, topo,            N, H=0) = 1
@inline total_length(::Type{Nothing}, ::Type{Flat},    N, H=0) = N
@inline total_length(::Type{Face},    ::Type{Flat},    N, H=0) = N
@inline total_length(::Type{Center},  ::Type{Flat},    N, H=0) = N

# Grid domains
@inline domain(topo, N, ξ) = CUDA.@allowscalar ξ[1], ξ[N+1]
@inline domain(::Type{Flat}, N, ξ) = CUDA.@allowscalar ξ[1], ξ[1]

@inline x_domain(grid) = domain(topology(grid, 1), grid.Nx, grid.xᶠᵃᵃ)
@inline y_domain(grid) = domain(topology(grid, 2), grid.Ny, grid.yᵃᶠᵃ)
@inline z_domain(grid) = domain(topology(grid, 3), grid.Nz, grid.zᵃᵃᶠ)

regular_dimensions(grid) = ()

#####
##### << Indexing >>
#####

@inline left_halo_indices(loc, topo, N, H) = 1-H:0
@inline left_halo_indices(::Type{Nothing}, topo, N, H) = 1:0 # empty

@inline right_halo_indices(loc, topo, N, H) = N+1:N+H
@inline right_halo_indices(::Type{Face}, ::Type{<:BoundedTopology}, N, H) = N+2:N+1+H
@inline right_halo_indices(::Type{Nothing}, topo, N, H) = 1:0 # empty

@inline underlying_left_halo_indices(loc, topo, N, H) = 1:H
@inline underlying_left_halo_indices(::Type{Nothing}, topo, N, H) = 1:0 # empty

@inline underlying_right_halo_indices(loc, topo, N, H) = N+1+H:N+2H
@inline underlying_right_halo_indices(::Type{Face}, ::Type{<:BoundedTopology}, N, H) = N+2+H:N+1+2H
@inline underlying_right_halo_indices(::Type{Nothing}, topo, N, H) = 1:0 # empty

@inline interior_indices(loc,             topo,            N) = 1:N
@inline interior_indices(::Type{Face},    ::Type{<:BoundedTopology}, N) = 1:N+1
@inline interior_indices(::Type{Nothing}, topo,            N) = 1:1

@inline interior_indices(::Type{Nothing}, topo::Type{Flat}, N) = 1:N
@inline interior_indices(::Type{Face},    topo::Type{Flat}, N) = 1:N
@inline interior_indices(::Type{Center},  topo::Type{Flat}, N) = 1:N

@inline interior_x_indices(loc, grid) = interior_indices(loc, topology(grid, 1), grid.Nx)
@inline interior_y_indices(loc, grid) = interior_indices(loc, topology(grid, 2), grid.Ny)
@inline interior_z_indices(loc, grid) = interior_indices(loc, topology(grid, 3), grid.Nz)

@inline interior_parent_offset(loc, topo, H) = H
@inline interior_parent_offset(::Type{Nothing}, topo, H) = 0

#@inline interior_parent_offset(::Type{Face},    topo, H) = H
# @inline interior_parent_offset(loc,             ::Type{Flat}, H) = 0
# @inline interior_parent_offset(::Type{Face},    ::Type{Flat}, H) = 0
#@inline interior_parent_offset(::Type{Nothing}, ::Type{Flat}, H) = 0

@inline interior_parent_indices(loc,             topo,            N, H) = 1+H:N+H
@inline interior_parent_indices(::Type{Face},    ::Type{<:BoundedTopology}, N, H) = 1+H:N+1+H
@inline interior_parent_indices(::Type{Nothing}, topo,            N, H) = 1:1

@inline interior_parent_indices(::Type{Nothing}, ::Type{Flat}, N, H) = 1:N
@inline interior_parent_indices(::Type{Face},    ::Type{Flat}, N, H) = 1:N
@inline interior_parent_indices(::Type{Center},  ::Type{Flat}, N, H) = 1:N

# All indices including halos.
@inline all_indices(loc,             topo,            N, H) = 1-H:N+H
@inline all_indices(::Type{Face},    ::Type{<:BoundedTopology}, N, H) = 1-H:N+1+H
@inline all_indices(::Type{Nothing}, topo,            N, H) = 1:1

@inline all_indices(::Type{Nothing}, ::Type{Flat}, N, H) = 1:N
@inline all_indices(::Type{Face},    ::Type{Flat}, N, H) = 1:N
@inline all_indices(::Type{Center},  ::Type{Flat}, N, H) = 1:N

@inline all_x_indices(loc, grid) = all_indices(loc, topology(grid, 1), grid.Nx, grid.Hx)
@inline all_y_indices(loc, grid) = all_indices(loc, topology(grid, 2), grid.Ny, grid.Hy)
@inline all_z_indices(loc, grid) = all_indices(loc, topology(grid, 3), grid.Nz, grid.Hz)

@inline all_parent_indices(loc,             topo,            N, H) = 1:N+2H
@inline all_parent_indices(::Type{Face},    ::Type{<:BoundedTopology}, N, H) = 1:N+1+2H
@inline all_parent_indices(::Type{Nothing}, topo,            N, H) = 1:1

@inline all_parent_indices(::Type{Nothing}, ::Type{Flat}, N, H) = 1:N
@inline all_parent_indices(::Type{Face},    ::Type{Flat}, N, H) = 1:N
@inline all_parent_indices(::Type{Center},  ::Type{Flat}, N, H) = 1:N

@inline all_parent_x_indices(loc, grid) = all_parent_indices(loc, topology(grid, 1), grid.Nx, grid.Hx)
@inline all_parent_y_indices(loc, grid) = all_parent_indices(loc, topology(grid, 2), grid.Ny, grid.Hy)
@inline all_parent_z_indices(loc, grid) = all_parent_indices(loc, topology(grid, 3), grid.Nz, grid.Hz)

parent_index_range(::Colon,                       loc, topo, halo) = Colon()
parent_index_range(::Base.Slice{<:IdOffsetRange}, loc, topo, halo) = Colon()
parent_index_range(index::UnitRange,              loc, topo, halo) = index .+ interior_parent_offset(loc, topo, halo)

parent_index_range(index::UnitRange, ::Type{Nothing}, ::Type{Flat}, halo) = index
parent_index_range(index::UnitRange, ::Type{Nothing},         topo, halo) = 1:1 # or Colon()

index_range_offset(index::UnitRange, loc, topo, halo) = index[1] - interior_parent_offset(loc, topo, halo)
index_range_offset(::Colon, loc, topo, halo)          = - interior_parent_offset(loc, topo, halo)

#####
##### << Nodes >>
#####

# Fallback
@inline xnode(LX, LY, LZ, i, j, k, grid) = xnode(LX, i, grid)
@inline ynode(LX, LY, LZ, i, j, k, grid) = ynode(LY, j, grid)
@inline znode(LX, LY, LZ, i, j, k, grid) = znode(LZ, k, grid)

@inline node(LX, LY, LZ, i, j, k, grid) = (xnode(LX, LY, LZ, i, j, k, grid),
                                           ynode(LX, LY, LZ, i, j, k, grid),
                                           znode(LX, LY, LZ, i, j, k, grid))

@inline node(LX::Nothing, LY, LZ, i, j, k, grid) = (ynode(LX, LY, LZ, i, j, k, grid), znode(LX, LY, LZ, i, j, k, grid))
@inline node(LX, LY::Nothing, LZ, i, j, k, grid) = (xnode(LX, LY, LZ, i, j, k, grid), znode(LX, LY, LZ, i, j, k, grid))
@inline node(LX, LY, LZ::Nothing, i, j, k, grid) = (xnode(LX, LY, LZ, i, j, k, grid), ynode(LX, LY, LZ, i, j, k, grid))

@inline node(LX, LY::Nothing, LZ::Nothing, i, j, k, grid) = tuple(xnode(LX, LY, LZ, i, j, k, grid))
@inline node(LX::Nothing, LY, LZ::Nothing, i, j, k, grid) = tuple(ynode(LX, LY, LZ, i, j, k, grid))
@inline node(LX::Nothing, LY::Nothing, LZ, i, j, k, grid) = tuple(znode(LX, LY, LZ, i, j, k, grid))

@inline cpu_face_constructor_x(grid) = Array(all_x_nodes(Face, grid)[1:grid.Nx+1])
@inline cpu_face_constructor_y(grid) = Array(all_y_nodes(Face, grid)[1:grid.Ny+1])
@inline cpu_face_constructor_z(grid) = Array(all_z_nodes(Face, grid)[1:grid.Nz+1])

all_x_nodes(::Type{Nothing}, grid) = 1:1
all_y_nodes(::Type{Nothing}, grid) = 1:1
all_z_nodes(::Type{Nothing}, grid) = 1:1

"""
    xnodes(loc, grid, reshape=false)

Return a view over the interior `loc=Center` or `loc=Face` nodes
on `grid` in the ``x``-direction. For `Bounded` directions,
`Face` nodes include the boundary points.

Keyword argument
================
- `reshape`: With `reshape=false` (default) the output is a 1D array while with 
  `reshape=true` the output is a 3D array with size `Nx×1×1`.

See `znodes` for examples.
"""
function xnodes(loc, grid; reshape=false)

    x = view(all_x_nodes(loc, grid),
             interior_indices(loc, topology(grid, 1), grid.Nx))

    return reshape ? Base.reshape(x, length(x), 1, 1) : x
end

"""
    ynodes(loc, grid, reshape=false)

Return a view over the interior `loc=Center` or `loc=Face` nodes
on `grid` in the ``y``-direction. For `Bounded` directions,
`Face` nodes include the boundary points.

Keyword argument
================
- `reshape`: With `reshape=false` (default) the output is a 1D array while with 
  `reshape=true` the output is a 3D array with size `1×Ny×1`.

See [`znodes`](@ref) for examples.
"""
function ynodes(loc, grid; reshape=false)

    y = view(all_y_nodes(loc, grid),
             interior_indices(loc, topology(grid, 2), grid.Ny))

    return reshape ? Base.reshape(y, 1, length(y), 1) : y
end

"""
    znodes(loc, grid, reshape=false)

Return a view over the interior `loc=Center` or `loc=Face` nodes
on `grid` in the ``z``-direction. For `Bounded` directions,
`Face` nodes include the boundary points.

Keyword argument
================
- `reshape`: With `reshape=false` (default) the output is a 1D array while with 
  `reshape=true` the output is a 3D array with size `1×1×Nz`.

Examples
========

```jldoctest znodes
julia> using Oceananigans

julia> horz_periodic_grid = RectilinearGrid(size=(3, 3, 3), extent=(2π, 2π, 1), halo=(1, 1, 1),
                                                 topology=(Periodic, Periodic, Bounded));

julia> zC = znodes(Center, horz_periodic_grid)
3-element view(OffsetArray(::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, 0:4), 1:3) with eltype Float64:
 -0.8333333333333334
 -0.5
 -0.16666666666666666
```

``` jldoctest znodes
julia> zF = znodes(Face, horz_periodic_grid)
4-element view(OffsetArray(::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, 0:5), 1:4) with eltype Float64:
 -1.0
 -0.6666666666666666
 -0.3333333333333333
  0.0
```
"""
function znodes(loc, grid; reshape=false)

    z = view(all_z_nodes(loc, grid),
             interior_indices(loc, topology(grid, 3), grid.Nz))

    return reshape ? Base.reshape(z, 1, 1, length(z)) : z
end

"""
    nodes(loc, grid; reshape=false)

Return a 3-tuple of views over the interior nodes
at the locations in `loc` in `x, y, z`.

If `reshape=true`, the views are reshaped to 3D arrays
with non-singleton dimensions 1, 2, 3 for `x, y, z`, respectively.
These reshaped arrays can then be used in broadcast operations with 3D fields
or arrays.

See [`xnodes`](@ref), [`ynodes`](@ref), and [`znodes`](@ref).
"""
function nodes(loc, grid::AbstractGrid; reshape=false)
    if reshape
        x, y, z = nodes(loc, grid; reshape=false)

        N = (length(x), length(y), length(z))

        x = Base.reshape(x, N[1], 1, 1)
        y = Base.reshape(y, 1, N[2], 1)
        z = Base.reshape(z, 1, 1, N[3])

        return (x, y, z)
    else
        return (xnodes(loc[1], grid),
                ynodes(loc[2], grid),
                znodes(loc[3], grid))
    end
end

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
Base.summary(::ZDirection) = "ZDirection"

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

It has been known since Euler and Lagrange that ``\\tan(E/2) = P / (1 + \\cos a + \\cos b + \\cos c)``, where
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
that ``P`` above  the same as the volume defined by the vectors `a`, `b`, and `c`, that is
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
overlapping, we compute the area of the quadrilateral as half the sum of the areas of all four potential
triangles.
"""
spherical_area_quadrilateral(a::AbstractVector, b::AbstractVector, c::AbstractVector, d::AbstractVector) =
    1/2 * (spherical_area_triangle(a, b, c) + spherical_area_triangle(a, b, d) +
           spherical_area_triangle(a, c, d) + spherical_area_triangle(b, c, d))

"""
    hav(x)

Compute haversine of `x`, where `x` is in radians: `hav(x) = sin²(x/2)`.
"""
hav(x) = sin(x/2)^2

"""
    central_angle((φ₁, λ₁), (φ₂, λ₂))

Compute the central angle (in radians) between two points on the sphere with
`(latitude, longitude)` coordinates `(φ₁, λ₁)` and `(φ₂, λ₂)` (in radians).

References
==========
- [Wikipedia, Great-circle distance](https://en.wikipedia.org/wiki/Great-circle_distance)
"""
function central_angle((φ₁, λ₁), (φ₂, λ₂))
    Δφ, Δλ = φ₁ - φ₂, λ₁ - λ₂

    return 2asin(sqrt(hav(Δφ) + (1 - hav(Δφ) - hav(φ₁ + φ₂)) * hav(Δλ)))
end

"""
    central_angle_degrees((φ₁, λ₁), (φ₂, λ₂))

Compute the central angle (in degrees) between two points on the sphere with
`(latitude, longitude)` coordinates `(φ₁, λ₁)` and `(φ₂, λ₂)` (in degrees).

See also [`central_angle`](@ref).
"""
central_angle_degrees((φ₁, λ₁), (φ₂, λ₂)) = rad2deg(central_angle(deg2rad.((φ₁, λ₁)), deg2rad.((φ₂, λ₂))))
