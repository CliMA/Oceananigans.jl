"""
    validate_grid(a::AbstractField, b::AbstractField)

Confirm that `a` and `b` are on the same grid if both are fields and return `a.grid`.
"""
function validate_grid(a::AbstractField, b::AbstractField)
    a.grid == b.grid || throw(ArgumentError("Fields in an AbstractOperation must be on the same grid."))
    return a.grid
end

"""Return `a.grid` when `b` is not an `AbstractField`."""
validate_grid(a::AbstractField, b) = a.grid

"""Return `b.grid` when `a` is not an `AbstractField`."""
validate_grid(a, b::AbstractField) = b.grid

"""Fallback when neither `a` nor `b` are `AbstractField`s."""
validate_grid(a, b) = nothing

"""
    validate_grid(a, b, c...)

Confirm that the grids associated with the 3+ long list `a, b, c...` are
consistent by checking each member against `a`.
This function is only correct when `a` is an `AbstractField`, though the
subsequent members `b, c...` may be anything.
"""
function validate_grid(a, b, c, d...)
    grids = []
    push!(grids, validate_grid(a, b))
    push!(grids, validate_grid(a, c))
    append!(grids, [validate_grid(a, di) for di in d])

    for g in grids
        if !(g === nothing)
            return g
        end
    end

    return nothing
end
