update_boundary_condition!(bc::Union{BoundaryCondition, Nothing}, args...; kwargs...) = nothing

function update_boundary_conditions!(bcs::FieldBoundaryConditions, field, model; kwargs...)
    update_boundary_condition!(bcs.west, field, model, Val(:west); kwargs...)
    update_boundary_condition!(bcs.east, field, model, Val(:east); kwargs...)
    update_boundary_condition!(bcs.south, field, model, Val(:south); kwargs...)
    update_boundary_condition!(bcs.north, field, model, Val(:north); kwargs...)
    update_boundary_condition!(bcs.bottom, field, model, Val(:bottom); kwargs...)
    update_boundary_condition!(bcs.top, field, model, Val(:top); kwargs...)
    update_boundary_condition!(bcs.immersed, field, model, Val(:immersed); kwargs...)
end

update_boundary_conditions!(fields::Union{NamedTuple, Tuple}, model; kwargs...) = 
    Tuple(update_boundary_conditions!(get_boundary_conditions(field), field, model; kwargs...) for field in fields)

@inline get_boundary_conditions(field) = nothing