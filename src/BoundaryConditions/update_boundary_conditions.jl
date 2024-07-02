update_boundary_conditions!(bc::Union{BoundaryCondition, Nothing}, args...) = nothing

function update_boundary_conditions!(bcs::FieldBoundaryConditions, field, model)
    update_boundary_conditions!(bcs.west, field, model, Val(:west))
    update_boundary_conditions!(bcs.east, field, model, Val(:east))
    update_boundary_conditions!(bcs.south, field, model, Val(:south))
    update_boundary_conditions!(bcs.north, field, model, Val(:north))
    update_boundary_conditions!(bcs.bottom, field, model, Val(:bottom))
    update_boundary_conditions!(bcs.top, field, model, Val(:top))
    update_boundary_conditions!(bcs.immersed, field, model, Val(:immersed))
end

update_boundary_conditions!(fields::Union{NamedTuple, Tuple}, model) = 
    Tuple(update_boundary_conditions!(get_boundary_conditions(field), field, model) for field in fields)

get_boundary_conditions(field) = nothing