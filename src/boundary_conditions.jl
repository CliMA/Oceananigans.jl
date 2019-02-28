struct BoundaryConditions
    x_bc::Symbol
    y_bc::Symbol
    top_bc::Symbol
    bottom_bc::Symbol

    function BoundaryConditions(x_bc, y_bc, top_bc, bottom_bc)
        @assert x_bc == :periodic && y_bc == :periodic "Only periodic horizontal boundary conditions are currently supported."
        @assert top_bc == :rigid_lid "Only rigid lid is currently supported at the top."
        @assert bottom_bc in [:no_slip, :free_slip] "Bottom boundary condition must be :no_slip or :free_slip."
        new(x_bc, y_bc, top_bc, bottom_bc)
    end
end
