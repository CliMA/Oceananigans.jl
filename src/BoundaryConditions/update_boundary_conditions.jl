using Oceananigans: boundary_conditions

update_boundary_condition!(bc::Union{BoundaryCondition, Nothing}, args...) = nothing

function update_boundary_condition!(bcs::FieldBoundaryConditions, field, model)
    update_boundary_condition!(bcs.west, Val(:west), field, model)
    update_boundary_condition!(bcs.east, Val(:east), field, model)
    update_boundary_condition!(bcs.south, Val(:south), field, model)
    update_boundary_condition!(bcs.north, Val(:north), field, model)
    update_boundary_condition!(bcs.bottom, Val(:bottom), field, model)
    update_boundary_condition!(bcs.top, Val(:top), field, model)
    update_boundary_condition!(bcs.immersed, Val(:immersed), field, model)
end

update_boundary_condition!(fields::Union{NamedTuple, Tuple}, model) = 
    Tuple(update_boundary_condition!(boundary_conditions(field), field, model) for field in fields)