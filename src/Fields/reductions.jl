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

struct Reducing end
struct Accumulating end

Base.summary(::Reducing) = "Reducing"
Base.summary(::Accumulating) = "Accumulating"

const Reduction = Scan{<:Reducing}
const Accumulation = Scan{<:Accumulating}

scan_indices(::Reducing, indices; dims) = Tuple(i ∈ dims ? Colon() : indices[i] for i in 1:3)
scan_indices(::Accumulating, indices; dims) = indices

Base.summary(r::Scan) = string(summary(s.type), " ",
                               r.scan!, 
                               " over dims ", r.dims,
                               " of ", summary(r.operand))

function Field(scan::Scan;
               data = nothing,
               indices = indices(scan.operand),
               recompute_safely = true)

    operand = scan.operand
    grid = operand.grid
    LX, LY, LZ = loc = location(scan)
    indices = scan_indices(indices; dims=scan.dims)

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
    s.scan!(field, s.operand)
    return field
end

#####
##### show
#####

Base.show(io::IO, field::ScannedComputedField) =
    print(io, "$(summary(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field))\n",
          "├── grid: $(summary(field.grid))\n",
          "├── operand: $(summary(field.operand))\n",
          "└── status: $(summary(field.status))")

Base.show(io::IO, r::Scan) =
    print(io, "$(summary(r))\n",
          "└── operand: $(summary(r.operand))\n",
          "    └── grid: $(summary(r.operand.grid))")

#####
##### Reductions (where the output has fewer dimensions than the input)
#####

"""
    Reduction(reduce!, operand; dims)

Return a `Reduction` of `operand` with `reduce!`, where `reduce!` can be called with

```
reduce!(field, operand; dims)
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
Accumulation(accumulate!, operand; dims) = Scan(Accumulating(), reduce!, operand, dims)
location(a::Accumulation) = location(a.operand)

