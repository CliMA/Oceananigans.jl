#####
##### Reductions of AbstractField
#####

struct Reduction{R, O, C, M, D}
    reduce! :: R
    operand :: O
    condition :: C
    mask :: M
    dims :: D
end

"""
    Reduction(reduce!, operand; dims)

Return a `Reduction` of `operand` with `reduce!`, along `dims`.

Example
=======

julia> grid = RectilinearGrid(size=(3, 3, 3), x=(0, 1), y=(0, 1), z=(0, 1),
                              topology=(Periodic, Periodic, Periodic))

julia> c = CenterField(grid)

julia> set!(c, (x, y, z) -> x + y + z)

julia> max_c² = Field(Reduction(maximum!, c^2, dims=3))

julia> compute!(max_c²)
"""
Reduction(reduce!, operand; condition, mask, dims) = Reduction(reduce!, operand, condition, mask, dims)

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
    reduction.reduce!(field, reduction.operand; condition = reduction.condition, mask = reduction.mask)
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
          "└── status: ", field.status)

Base.summary(r::Reduction) = string(r.reduce!, 
                                    " over dims ", r.dims,
                                    " of ", summary(r.operand))
