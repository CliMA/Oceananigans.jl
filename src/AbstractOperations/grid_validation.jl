function validate_grid(a::AbstractField, b::AbstractField)
    a.grid === b.grid || throw(ArgumentError("Two fields in a BinaryOperation must be on the same grid."))
    return a.grid
end

validate_grid(a::AbstractField, b) = a.grid
validate_grid(a, b::AbstractField) = b.grid
validate_grid(a, b) = nothing

function validate_grid(a, b, c...)
    grids = []
    push!(grids, validate_grid(a, b))
    append!(grids, [validate_grid(a, ci) for ci in c])

    for g in grids
        if !(g === nothing)
            return g
        end
    end

    return nothing
end

