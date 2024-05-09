using KernelAbstractions: @kernel, @index

#####
##### "Scans" of AbstractField.
#####

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

scan_indices(::AbstractReducing, indices; dims) = Tuple(i ∈ dims ? Colon() : indices[i] for i in 1:3)
scan_indices(::AbstractAccumulating, indices; dims) = indices

Base.summary(s::Scan) = string(summary(s.type), " ",
                               s.scan!, 
                               " over dims ", s.dims,
                               " of ", summary(s.operand))

function Field(scan::Scan;
               data = nothing,
               indices = indices(scan.operand),
               recompute_safely = true)

    operand = scan.operand
    grid = operand.grid
    LX, LY, LZ = loc = location(scan)
    indices = scan_indices(scan.type, indices; dims=scan.dims)

    if isnothing(data)
        data = new_data(grid, loc, indices)
        recompute_safely = false
    end

    boundary_conditions = FieldBoundaryConditions(grid, loc, indices)
    status = recompute_safely ? nothing : FieldStatus()

    return Field(loc, grid, data, boundary_conditions, indices, scan, status)
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

```
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

compute!(max_c²)

max_c²[1:Nx, 1:Ny]

# output
3×3 Matrix{Float64}:
 1.36111  2.25     3.36111
 2.25     3.36111  4.69444
 3.36111  4.69444  6.25
```
"""
Reduction(reduce!, operand; dims) = Scan(Reducing(), reduce!, operand, dims)
location(r::Reduction) = reduced_location(location(r.operand); dims=r.dims)

#####
##### Accumulations (where the output has the same dimensions as the input)
#####

"""
    Accumulation(accumulate!, operand; dims)

Return a `Accumulation` of `operand` with `accumulate!`, where `accumulate!` can be called with

```
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

max_c² = Field(Accumulation(cumsum!, c^2, dims=3))

compute!(max_c²)

max_c²[1:Nx, 1:Ny]

# output
3×3 Matrix{Float64}:
 1.36111  2.25     3.36111
 2.25     3.36111  4.69444
 3.36111  4.69444  6.25
```
"""
Accumulation(accumulate!, operand; dims) = Scan(Accumulating(), accumulate!, operand, dims)
location(a::Accumulation) = location(a.operand)

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
Base.accumulate!(op, B::Field, A::AbstractField; dims::Integer) =
    directional_accumulate!(op, B, A, dims, Forward())

reverse_accumulate!(op, B::Field, A::AbstractField; dims::Integer) =
    directional_accumulate!(op, B, A, dims, Reverse())

Base.cumsum!(B::Field, A::AbstractField; dims) =
    directional_accumulate!(Base.add_sum, B, A, dims, Forward())

reverse_cumsum!(B::Field, A::AbstractField; dims) =
    directional_accumulate!(Base.add_sum, B, A, dims, Reverse())

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

    launch!(arch, grid, config, kernel, op, B, A, start, finish, direction)

    return B
end

@inline function accumulation_range(dir, start, finish)
    by = increment(dir, 0)
    from = increment(dir, start)
    return StepRange(from, by, finish)
end

@kernel function accumulate_x(op, B, A, start, finish, dir)
    j, k = @index(Global, NTuple)

    # Initialize
    @inbounds B[start, j, k] = Base.reduce_first(op, A[start, j, k])

    for i in accumulation_range(dir, start, finish)
        pr = decrement(dir, i)
        @inbounds B[i, j, k] = op(B[pr, j, k], A[i, j, k])
    end
end

@kernel function accumulate_y(op, B, A, start, finish, dir)
    i, k = @index(Global, NTuple)

    # Initialize
    @inbounds B[i, start, k] = Base.reduce_first(op, A[i, start, k])

    for j in accumulation_range(dir, start, finish)
        pr = decrement(dir, j)
        @inbounds B[i, j, k] = op(B[i, pr, k], A[i, j, k])
    end
end

@kernel function accumulate_z(op, B, A, start, finish, dir)
    i, j = @index(Global, NTuple)

    # Initialize
    @inbounds B[i, j, start] = Base.reduce_first(op, A[i, j, start])

    for k in accumulation_range(dir, start, finish)
        pr = decrement(dir, k)
        @inbounds B[i, j, k] = op(B[i, j, pr], A[i, j, k])
    end
end

