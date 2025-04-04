using Oceananigans: boundary_conditions

@inline update_boundary_condition!(bc, side, field, model) = nothing
@inline update_boundary_condition!(bcs, field, model) = nothing

function update_boundary_condition!(bcs::FieldBoundaryConditions, field, model)
    update_boundary_condition!(bcs.west, Val(:west), field, model)
    update_boundary_condition!(bcs.east, Val(:east), field, model)
    update_boundary_condition!(bcs.south, Val(:south), field, model)
    update_boundary_condition!(bcs.north, Val(:north), field, model)
    update_boundary_condition!(bcs.bottom, Val(:bottom), field, model)
    update_boundary_condition!(bcs.top, Val(:top), field, model)
    update_boundary_condition!(bcs.immersed, Val(:immersed), field, model)
    return nothing
end

update_boundary_condition!(fields::NamedTuple, model) = update_boundary_condition!(values(fields), model)

function update_boundary_condition!(fields::Tuple, model)
    for field in fields
        bcs = boundary_conditions(field)
        update_boundary_condition!(bcs, field, model)
    end

    return nothing
end
