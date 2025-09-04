using Base.Ryu: writeshortest
using LinearAlgebra: dot, cross
using OffsetArrays: IdOffsetRange

"""
    _property(ξ, T, ℓ, N, with_halos)

Return the grid property `ξ`, either `with_halos` or without,
for topology `T`, (instantiated) location `ℓ`, and dimension length `N`.
"""
@inline function _property(ξ, ℓ, T, N, H, with_halos)
    if with_halos
        return ξ
    else
        i = interior_parent_indices(ℓ, T(), N, H)
        return view(parent(ξ), i)
    end
end

@inline function _property(ξ, ℓx, ℓy, Tx, Ty, Nx, Ny, Hx, Hy, with_halos)
    if with_halos
        return ξ
    else
        i = interior_parent_indices(ℓx, Tx(), Nx, Hx)
        j = interior_parent_indices(ℓy, Ty(), Ny, Hy)
        return view(parent(ξ), i, j)
    end
end

@inline _property(ξ::Number, args...) = ξ
@inline _property(::Nothing, args...) = nothing

# Define default indices in a type-stable way
@inline default_indices(N::Int) = default_indices(Val(N))

@inline function default_indices(::Val{N}) where N
    ntuple(Val(N)) do n
        Base.@_inline_meta
        Colon()
    end
end

const BoundedTopology = Union{Bounded, LeftConnected}
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
Base.length(loc, topo::AT, N, ind::AbstractUnitRange) = min(length(loc, topo, N), length(ind))

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
total_length(loc, topo, N, H, ind::AbstractUnitRange)  = min(total_length(loc, topo, N, H), length(ind))

@inline Base.size(grid::AbstractGrid, loc::Tuple, indices=default_indices(Val(length(loc)))) =
    size(loc, topology(grid), size(grid), indices)

@inline function Base.size(loc, topo, sz, indices=default_indices(Val(length(loc))))
    D = length(loc)

    # (it's type stable?)
    return ntuple(Val(D)) do d
        Base.@_inline_meta
        length(instantiate(loc[d]), instantiate(topo[d]), sz[d], indices[d])
    end
end

Base.size(grid::AbstractGrid, loc::Tuple, d::Int) = size(grid, loc)[d]

total_size(a) = size(a) # fallback

"""
    total_size(grid, loc)

Return the "total" size of a `grid` at `loc`. This is a 3-tuple of integers
corresponding to the number of grid points along `x, y, z`.
"""
function total_size(loc, topo, sz, halo_sz, indices=default_indices(Val(length(loc))))
    D = length(loc)
    N = ntuple(Val(D)) do d
        Base.@_inline_meta
        @inbounds total_length(instantiate(loc[d]), instantiate(topo[d]), sz[d], halo_sz[d], indices[d])
    end
    return N
end

total_size(grid::AbstractGrid, loc, indices=default_indices(Val(length(loc)))) =
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
@inline domain(topo, N, ξ) = @allowscalar ξ[1], ξ[N+1]
@inline domain(::Flat, N, ξ::AbstractArray) = ξ[1]
@inline domain(::Flat, N, ξ::Number) = ξ
@inline domain(::Flat, N, ::Nothing) = nothing

@inline x_domain(grid) = domain(topology(grid, 1)(), grid.Nx, grid.xᶠᵃᵃ)
@inline y_domain(grid) = domain(topology(grid, 2)(), grid.Ny, grid.yᵃᶠᵃ)

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

# Return the index range of "full" parent arrays that span an entire dimension
parent_index_range(::Colon,                       loc, topo, halo) = Colon()
parent_index_range(::Base.Slice{<:IdOffsetRange}, loc, topo, halo) = Colon()
parent_index_range(view_indices::AbstractUnitRange, ::Nothing, ::Flat, halo) = view_indices
parent_index_range(view_indices::AbstractUnitRange, ::Nothing, ::AT,   halo) = 1:1 # or Colon()
parent_index_range(view_indices::AbstractUnitRange, loc, topo, halo) = view_indices .+ interior_parent_offset(loc, topo, halo)

# Return the index range of parent arrays that are themselves windowed
parent_index_range(::Colon, args...) = parent_index_range(args...)

parent_index_range(parent_indices::AbstractUnitRange, ::Colon, args...) =
    parent_index_range(parent_indices, parent_indices, args...)

function parent_index_range(parent_indices::AbstractUnitRange, view_indices, args...)
    start = first(view_indices) - first(parent_indices) + 1
    stop = start + length(view_indices) - 1
    return UnitRange(start, stop)
end

# intersect_index_range(::Colon, ::Colon) = Colon()
index_range_contains(range, subset::AbstractUnitRange) = (first(subset) ∈ range) & (last(subset) ∈ range)
index_range_contains(::Colon, ::AbstractUnitRange)     = true
index_range_contains(::Colon, ::Colon)                 = true
index_range_contains(::AbstractUnitRange, ::Colon)     = true

# Return the index range of "full" parent arrays that span an entire dimension
parent_windowed_indices(::Colon, loc, topo, halo)             = Colon()
parent_windowed_indices(indices::AbstractUnitRange, loc, topo, halo) = UnitRange(1, length(indices))

index_range_offset(index::AbstractUnitRange, loc, topo, halo) = index[1] - interior_parent_offset(loc, topo, halo)
index_range_offset(::Colon, loc, topo, halo)           = - interior_parent_offset(loc, topo, halo)

const c = Center()
const f = Face()

# What's going on here?
@inline cpu_face_constructor_x(grid) = Array(getindex(nodes(grid, f, c, c; with_halos=true), 1)[1:size(grid, 1)+1])
@inline cpu_face_constructor_y(grid) = Array(getindex(nodes(grid, c, f, c; with_halos=true), 2)[1:size(grid, 2)+1])

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

-(::NegativeZDirection) = ZDirection()
-(::ZDirection) = NegativeZDirection()

#####
##### Show utils
#####

Base.summary(::XDirection) = "XDirection()"
Base.summary(::YDirection) = "YDirection()"
Base.summary(::ZDirection) = "ZDirection()"
Base.summary(::NegativeZDirection) = "NegativeZDirection()"

Base.show(io::IO, dir::AbstractDirection) = print(io, summary(dir))

size_summary(sz) = string(sz[1], "×", sz[2], "×", sz[3])
prettysummary(σ::AbstractFloat, plus=false) = writeshortest(σ, plus, false, true, -1, UInt8('e'), false, UInt8('.'), false, true)

domain_summary(topo::Flat, name, ::Nothing) = "Flat $name"
domain_summary(topo::Flat, name, coord::Number) = "Flat $name = $coord"

function domain_summary(topo, name, (left, right))
    interval = (topo isa Bounded) ||
               (topo isa LeftConnected) ? "]" : ")"

    topo_string = topo isa Periodic ? "Periodic " :
                  topo isa Bounded ? "Bounded  " :
                  topo isa FullyConnected ? "FullyConnected " :
                  topo isa LeftConnected ? "LeftConnected  " :
                  topo isa RightConnected ? "RightConnected  " :
                  error("Unexpected topology $topo together with the domain end points ($left, $right)")

    return string(topo_string, name, " ∈ [",
                  prettysummary(left), ", ",
                  prettysummary(right), interval)
end

function dimension_summary(topo, name, dom, spacing, pad_domain=0)
    prefix = domain_summary(topo, name, dom)
    padding = " "^(pad_domain+1)
    return string(prefix, padding, coordinate_summary(topo, spacing, name))
end

coordinate_summary(::Flat, Δ::Number, name) = ""
coordinate_summary(topo, Δ::Number, name) = @sprintf("regularly spaced with Δ%s=%s", name, prettysummary(Δ))

coordinate_summary(topo, Δ::Union{AbstractVector, AbstractMatrix}, name) =
    @sprintf("variably spaced with min(Δ%s)=%s, max(Δ%s)=%s",
             name, prettysummary(minimum(parent(Δ))),
             name, prettysummary(maximum(parent(Δ))))

#####
##### Static and Dynamic column depths
#####

@inline static_column_depthᶜᶜᵃ(i, j, grid) = grid.Lz
@inline static_column_depthᶜᶠᵃ(i, j, grid) = grid.Lz
@inline static_column_depthᶠᶜᵃ(i, j, grid) = grid.Lz
@inline static_column_depthᶠᶠᵃ(i, j, grid) = grid.Lz

# Will be extended in the `ImmersedBoundaries` module for a ``mutable'' grid type
@inline column_depthᶜᶜᵃ(i, j, k, grid, η) = static_column_depthᶜᶜᵃ(i, j, grid)
@inline column_depthᶠᶜᵃ(i, j, k, grid, η) = static_column_depthᶠᶜᵃ(i, j, grid)
@inline column_depthᶜᶠᵃ(i, j, k, grid, η) = static_column_depthᶜᶠᵃ(i, j, grid)
@inline column_depthᶠᶠᵃ(i, j, k, grid, η) = static_column_depthᶠᶠᵃ(i, j, grid)

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
    map(a -> on_architecture(CPU(), a), data) # bring to CPU

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
    map(a -> on_architecture(arch, a), offset_array)

    return offset_array
end

function add_halos(data::AbstractArray{FT, 2} where FT, loc, topo, sz, halo_sz; warnings=true)
    Nx, Ny = size(data)
    return add_halos(reshape(data, (Nx, Ny, 1)), loc, topo, sz, halo_sz; warnings)
end
