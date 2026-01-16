using KernelAbstractions: @kernel, @index

#####
##### "Scans" of AbstractField.
#####

filter_nothing_dims(::Colon, loc) = filter_nothing_dims((1, 2, 3), loc)
filter_nothing_dims(dims, loc) = filter(d -> !isnothing(loc[d]), dims)
filter_nothing_dims(dim::Int, loc) = filter_nothing_dims(tuple(dim), loc)

"""
    Scan{T, R, O, D}

An operand for `Field` that is computed by traversing a dimension.
This includes `Reducing` `Scan`s like `sum!` and `maximum!`, as well as
`Accumulating` `Scan`s like `cumsum!`.
"""
struct Scan{T, R, O, D}
    type :: T
    scan! :: R
    operand :: O
    dims :: D
end

abstract type AbstractReducing end
abstract type AbstractAccumulating end

struct Reducing <: AbstractReducing end
struct Accumulating <: AbstractAccumulating end

Base.summary(::Reducing) = "Reducing"
Base.summary(::Accumulating) = "Accumulating"

const Reduction = Scan{<:AbstractReducing}
const Accumulation = Scan{<:AbstractAccumulating}

scan_indices(::AbstractReducing, indices, dims) = Tuple(i ∈ dims ? Colon() : indices[i] for i in 1:3)
scan_indices(::AbstractAccumulating, indices, dims) = indices
scan_indices(::AbstractReducing, ::Tuple{Colon, Colon, Colon}, dims) = (:, :, :)

Base.summary(s::Scan) = string(summary(s.type), " ",
                               s.scan!,
                               " over dims ", s.dims,
                               " of ", summary(s.operand))

function Field(scan::Scan;
               data = nothing,
               indices = indices(scan.operand),
               compute = true,
               recompute_safely = true)

    operand = scan.operand
    grid = operand.grid
    LX, LY, LZ = loc = instantiated_location(scan)
    dims = filter_nothing_dims(scan.dims, loc)
    indices = scan_indices(scan.type, indices, dims)

    if isnothing(data)
        data = new_data(grid, loc, indices)
        recompute_safely = false
    end

    boundary_conditions = FieldBoundaryConditions(grid, loc, indices)
    status = recompute_safely ? nothing : FieldStatus()

    scan_field = Field(loc, grid, data, boundary_conditions, indices, scan, status)

    if compute
         compute!(scan_field)
    end

    return scan_field
end

const ScannedComputedField = Field{<:Any, <:Any, <:Any, <:Scan}

function compute!(field::ScannedComputedField, time=nothing)
    s = field.operand
    compute_at!(s.operand, time)

    if s.type isa AbstractReducing
        s.scan!(field, s.operand)
    elseif s.type isa AbstractAccumulating
        s.scan!(field, s.operand; dims=s.dims)
    end

    set_status!(field.status, time)

    return field
end

#####
##### show
#####

function Base.show(io::IO, field::ScannedComputedField)
    print(io, summary(field), '\n',
          "├── data: ", typeof(field.data), ", size: ", size(field), '\n',
          "├── grid: ", summary(field.grid), '\n',
          "├── operand: ", summary(field.operand), '\n',
          "├── status: ", summary(field.status), '\n')

    data_str = string("└── data: ", summary(field.data), '\n',
                      "    └── ", data_summary(field))

    print(io, data_str)
end

Base.show(io::IO, s::Scan) =
    print(io, "$(summary(s))\n",
          "└── operand: $(summary(s.operand))\n",
          "    └── grid: $(summary(s.operand.grid))")

#####
##### Reductions (where the output has fewer dimensions than the input)
#####

"""
    Reduction(reduce!, operand; dims)

Return a `Reduction` of `operand` with `reduce!`, where `reduce!` can be called with

```julia
reduce!(field, operand)
```

to reduce `operand` along `dims` and store in `field`.

Example
=======

```jldoctest
using Oceananigans

Nx, Ny, Nz = 3, 3, 3
grid = RectilinearGrid(size=(Nx, Ny, Nz), x=(0, 1), y=(0, 1), z=(0, 1),
                       topology=(Periodic, Periodic, Periodic))

c = CenterField(grid)
set!(c, (x, y, z) -> x + y + z)
max_c² = Field(Reduction(maximum!, c^2, dims=3))
max_c²[1:Nx, 1:Ny]

# output
3×3 Matrix{Float64}:
 1.36111  2.25     3.36111
 2.25     3.36111  4.69444
 3.36111  4.69444  6.25
```
"""
Reduction(reduce!, operand; dims) = Scan(Reducing(), reduce!, operand, dims)
Oceananigans.location(r::Reduction) = reduced_location(location(r.operand); dims=r.dims)

#####
##### Accumulations (where the output has the same dimensions as the input)
#####

"""
    Accumulation(accumulate!, operand; dims)

Return a `Accumulation` of `operand` with `accumulate!`, where `accumulate!` can be called with

```julia
accumulate!(field, operand; dims)
```

to accumulate `operand` along `dims` and store in `field`.

Example
=======

```jldoctest
using Oceananigans

Nx, Ny, Nz = 3, 3, 3
grid = RectilinearGrid(size=(Nx, Ny, Nz), x=(0, 1), y=(0, 1), z=(0, 1),
                       topology=(Periodic, Periodic, Periodic))

c = CenterField(grid)
set!(c, (x, y, z) -> x + y + z)
cumsum_c² = Field(Accumulation(cumsum!, c^2, dims=3))
cumsum_c²[1:Nx, 1:Ny, 1:Nz]

# output
3×3×3 Array{Float64, 3}:
[:, :, 1] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 2] =
 0.25      0.694444  1.36111
 0.694444  1.36111   2.25
 1.36111   2.25      3.36111

[:, :, 3] =
 0.944444  2.05556  3.61111
 2.05556   3.61111  5.61111
 3.61111   5.61111  8.05556
```
"""
Accumulation(accumulate!, operand; dims) = Scan(Accumulating(), accumulate!, operand, dims)

flip(::Type{Face}) = Center
flip(::Type{Center}) = Face

function Oceananigans.location(a::Accumulation)
    op_loc = location(a.operand)
    loc = Tuple(d ∈ a.dims ? flip(op_loc[d]) : op_loc[d] for d=1:3)
    return loc
end

#####
##### Some custom scans
#####

# directions
struct Forward end
struct Reverse end

@inline increment(::Forward, idx) = idx + 1
@inline decrement(::Forward, idx) = idx - 1

@inline increment(::Reverse, idx) = idx - 1
@inline decrement(::Reverse, idx) = idx + 1

# TODO: if there is a way to re-use Base stuff rather than write ourselves, that might be preferred.
# Also: these don't work on ImmersedBoundaryGrid out of the box,
# we need to somehow find the neutral element for `op`
Base.accumulate!(op, B::Field, A::AbstractField; dims::Integer) =
    directional_accumulate!(op, B, A, dims, Forward())

reverse_accumulate!(op, B::Field, A::AbstractField; dims::Integer) =
    directional_accumulate!(op, B, A, dims, Reverse())

function Base.cumsum!(B::Field, A::AbstractField; dims, condition=nothing, mask=get_neutral_mask(Base.sum!))
    Ac = condition_operand(A, condition, mask)
    return directional_accumulate!(Base.add_sum, B, Ac, dims, Forward())
end

function reverse_cumsum!(B::Field, A::AbstractField; dims, condition=nothing, mask=get_neutral_mask(Base.sum!))
    Ac = condition_operand(A, condition, mask)
    return directional_accumulate!(Base.add_sum, B, Ac, dims, Reverse())
end

function directional_accumulate!(op, B, A, dim, direction)

    grid = B.grid
    arch = architecture(B)

    # TODO: this won't work on windowed fields
    # To fix this we can change config, start, and finish.
    if dim == 1
        config = :yz
        kernel = accumulate_x
    elseif dim == 2
        config = :xz
        kernel = accumulate_y
    elseif dim == 3
        config = :xy
        kernel = accumulate_z
    end

    if direction isa Forward
        start = 1
        finish = size(B, dim)
    elseif direction isa Reverse
        start = size(B, dim)
        finish = 1
    end

    # Determine if we're "expanding" (output has more points than input)
    # This affects which A index to use for reverse accumulation
    expanding = size(B, dim) > size(A, dim)

    launch!(arch, grid, config, kernel, op, B, A, start, finish, direction, expanding)

    return B
end

@inline function accumulation_range(dir, start, finish)
    by = increment(dir, 0)
    from = increment(dir, start)
    return StepRange(from, by, finish)
end

# TODO: extend to more operators
neutral_element(::typeof(Base.min), T) = convert(T, +Inf)
neutral_element(::typeof(Base.max), T) = convert(T, -Inf)
neutral_element(::typeof(Base.add_sum), T) = convert(T, 0)

# For computing the correct A index when locations are flipped (Center ↔ Face):
# - Forward (all cases): use A[previous] (k-1)
# - Reverse expanding (Center→Face, more output points): use A[current] (k)
# - Reverse contracting (Face→Center, fewer output points): use A[previous] (k+1)
@inline accumulate_A_index(::Forward, current, previous, expanding) = previous
@inline accumulate_A_index(::Reverse, current, previous, expanding) = expanding ? current : previous

@kernel function accumulate_x(op, B, A, start, finish, dir, expanding)
    j, k = @index(Global, NTuple)

    # Initialize with neutral element
    FT = eltype(B)
    @inbounds B[start, j, k] = neutral_element(op, FT)

    # Accumulate with correct A index based on direction and size relationship
    for i in accumulation_range(dir, start, finish)
        pr = decrement(dir, i)
        Ai = accumulate_A_index(dir, i, pr, expanding)
        @inbounds B[i, j, k] = op(B[pr, j, k], A[Ai, j, k])
    end
end

@kernel function accumulate_y(op, B, A, start, finish, dir, expanding)
    i, k = @index(Global, NTuple)

    # Initialize with neutral element
    FT = eltype(B)
    @inbounds B[i, start, k] = neutral_element(op, FT)

    # Accumulate with correct A index based on direction and size relationship
    for j in accumulation_range(dir, start, finish)
        pr = decrement(dir, j)
        Aj = accumulate_A_index(dir, j, pr, expanding)
        @inbounds B[i, j, k] = op(B[i, pr, k], A[i, Aj, k])
    end
end

@kernel function accumulate_z(op, B, A, start, finish, dir, expanding)
    i, j = @index(Global, NTuple)

    # Initialize with neutral element
    FT = eltype(B)
    @inbounds B[i, j, start] = neutral_element(op, FT)

    # Accumulate with correct A index based on direction and size relationship
    for k in accumulation_range(dir, start, finish)
        pr = decrement(dir, k)
        Ak = accumulate_A_index(dir, k, pr, expanding)
        @inbounds B[i, j, k] = op(B[i, j, pr], A[i, j, Ak])
    end
end
