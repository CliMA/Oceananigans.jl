validate_boundary_condition(topo, bc, loc, side) = error("$side boundary condition $bc cannot be applied " *
                                                          "to a field located at $loc in a $topo dimension.")

# Whitelist...
CenterBoundedBCs = Union{BoundaryCondition{<:Value},
                         BoundaryCondition{<:Gradient},
                         BoundaryCondition{<:Flux}}

validate_boundary_condition(::Bounded, ::CenterBoundedBCs, ::Center, side) = nothing # fallback
validate_boundary_condition(::Bounded, ::Union{nothing, BoundaryCondition{<:NormalFlow}}, ::Face, side) = nothing # fallback
validate_boundary_condition(::Periodic, ::BoundaryCondition{<:Periodic}, loc, side) = nothing

# Validate boundary conditions
validate_field_boundary_conditions(::Nothing, grid, LX, LY, LZ) = nothing

function validate_field_boundary_conditions(bcs, grid, LX, LY, LZ)
    west_bc   = bcs.west
    east_bc   = bcs.east
    south_bc  = bcs.south
    north_bc  = bcs.north
    bottom_bc = bcs.bottom
    top_bc    = bcs.top

    TX, TY, TZ = topology(grid)

    validate_boundary_condition(TX, west_bc, LX, :west)
    validate_boundary_condition(TX, east_bc, LX, :east)

    validate_boundary_condition(TY, south_bc, LY, :south)
    validate_boundary_condition(TY, north_bc, LY, :north)

    validate_boundary_condition(TZ, top_bc, LZ, :top)
    validate_boundary_condition(TZ, bottom_bc, LZ, :bottom)

    return nothing
end

