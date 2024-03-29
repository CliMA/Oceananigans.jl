update_boundary_condition!(bc::Union{BoundaryCondition, Nothing}, args...) = nothing

function update_boundary_conditions!(bcs::FieldBoundaryConditions, field, model)
    update_boundary_condition!(bcs.west, field, model, Val(:west))
    update_boundary_condition!(bcs.east, field, model, Val(:east))
    update_boundary_condition!(bcs.south, field, model, Val(:south))
    update_boundary_condition!(bcs.north, field, model, Val(:north))
    update_boundary_condition!(bcs.bottom, field, model, Val(:bottom))
    update_boundary_condition!(bcs.top, field, model, Val(:top))
    update_boundary_condition!(bcs.immersed, field, model, Val(:immersed))
end

update_boundary_conditions!(fields::Union{NamedTuple, Tuple}, model) = 
    Tuple(update_boundary_conditions!(field.boundary_conditions, field, model) for field in fields)

