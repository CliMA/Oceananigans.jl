using Oceananigans: boundary_conditions

@inline update_boundary_conditions!(bcs, field, model) = nothing
@inline update_boundary_condition!(bc, side, field, model) = nothing

function update_boundary_conditions!(bcs::FieldBoundaryConditions, field, model)
    update_boundary_condition!(bcs.west, Val(:west), field, model)
    update_boundary_condition!(bcs.east, Val(:east), field, model)
    update_boundary_condition!(bcs.south, Val(:south), field, model)
    update_boundary_condition!(bcs.north, Val(:north), field, model)
    update_boundary_condition!(bcs.bottom, Val(:bottom), field, model)
    update_boundary_condition!(bcs.top, Val(:top), field, model)
    update_boundary_condition!(bcs.immersed, Val(:immersed), field, model)
    return nothing
end

@inline update_boundary_conditions!(fields::NamedTuple, model) = update_boundary_conditions!(values(fields), model)
@inline update_boundary_conditions!(::Tuple{},          model) = nothing

@inline function update_boundary_conditions!(fields::Tuple, model)
    field = first(fields)
    update_boundary_conditions!(boundary_conditions(field), field, model)
    update_boundary_conditions!(Base.tail(fields), model)
    return nothing
end
