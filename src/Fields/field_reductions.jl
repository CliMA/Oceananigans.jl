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
expects `reduce!` to operate in-place.

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

location(r::Reduction) = reduced_location(location(r.operand); dims=r.dims)

function Field(reduction::Reduction;
               data = nothing,
               indices = indices(reduction.operand),
               recompute_safely = false)

    operand = reduction.operand
    grid = operand.grid
    
    loc     = location(reduction)
    o_loc   = location(reduction.operand)
    def_idx = default_indices(loc, length(loc))
    
    idx = Tuple(loc[i] == o_loc[i] ? indices[i] : def_idx[i] for i in 1:length(loc))
    
    if isnothing(data)
        data = new_data(grid, loc, idx)
        recompute_safely = false
    end

    boundary_conditions = FieldBoundaryConditions(grid, loc, indices)
    status = recompute_safely ? nothing : FieldStatus()

    return Field(loc, grid, data, boundary_conditions, idx, reduction, status)
end

const ReducedComputedField = Field{<:Any, <:Any, <:Any, <:Reduction}

function compute!(field::ReducedComputedField, time=nothing)
    reduction = field.operand
    compute_at!(reduction.operand, time)
    reduction.reduce!(field, reduction.operand)
    return field
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

Base.show(io::IO, r::Reduction) =
    print(io, "$(summary(r))\n",
          "└── operand: $(summary(r.operand))\n",
          "    └── grid: $(summary(r.operand.grid))")
