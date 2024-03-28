update_boundary_condition!(::Union{BoundaryCondition, Nothing}, args...) = nothing

function update_boundary_conditions!(bcs::FieldBoundaryConditions, field, model)
    update_boundary_condition!(bcs.west, field, model)
    update_boundary_condition!(bcs.east, field, model)
    update_boundary_condition!(bcs.south, field, model)
    update_boundary_condition!(bcs.north, field, model)
    update_boundary_condition!(bcs.bottom, field, model)
    update_boundary_condition!(bcs.top, field, model)
    update_boundary_condition!(bcs.immersed, field, model)
end

update_boundary_conditions!(fields::Union{NamedTuple, Tuple}, model) = 
    Tuple(update_boundary_conditions!(field.boundary_conditions, field, model) for field in fields)

