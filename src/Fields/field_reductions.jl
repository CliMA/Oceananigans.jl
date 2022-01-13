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
    print(io, "$(short_show(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field))\n",
          "├── grid: $(short_show(field.grid))\n",
          "├── dims: $(field.dims)\n",
          "├── operand: $(short_show(field.operand))\n",
          "└── status: ", show_status(field.status))

short_show(field::ReducedComputedField) = string("Field at ", show_location(field), " via ", short_show(field.operand))

short_show(r::Reduction) = string(typeof(r.reduce!), " of ", short_show(r.operand),
                                  " over dims ", r.dims)
                                  
