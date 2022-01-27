#####
##### Reductions of AbstractField
#####

struct Reduction{R, O, D}
    reduce! :: R
    operand :: O
    dims :: D
end

"""
    Reduction(reduce!, operand; dims)

Return a `Reduction` of `operand` with `reduce!`, along `dims`. Note that `Reduction`
expects a `reduce!` operation that works in-place.

Field elements for which `condition(i ,j ,k) == true` are masked with a `mask`.

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
Reduction(reduce!, operand; dims) = Reduction(reduce!, operand, dims)

function Field(reduction::Reduction;
               data = nothing,
               recompute_safely = false)

    operand = reduction.operand
    grid = operand.grid
    LX, LY, LZ = loc = reduced_location(location(operand); dims=reduction.dims)

    if isnothing(data)
        data = new_data(grid, loc)
        recompute_safely = false
    end

    boundary_conditions = FieldBoundaryConditions(grid, loc)
    status = recompute_safely ? nothing : FieldStatus()

    return Field(loc, grid, data, boundary_conditions, reduction, status)
end

const ReducedComputedField = Field{<:Any, <:Any, <:Any, <:Reduction}

function compute!(field::ReducedComputedField, time=nothing)
    reduction = field.operand
    compute_at!(reduction.operand, time)
    reduction.reduce!(field, reduction.operand)
    return nothing
end

#####
##### show
#####

Base.show(io::IO, field::ReducedComputedField) =
    print(io, "$(summary(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field))\n",
          "├── grid: $(summary(field.grid))\n",
          "├── operand: $(summary(field.operand))\n",
          "└── status: $(summary(field.status))")

Base.summary(r::Reduction) = string(r.reduce!, 
                                    " over dims ", r.dims,
                                    " of ", summary(r.operand))
