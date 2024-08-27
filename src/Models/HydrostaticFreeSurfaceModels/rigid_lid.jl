#####
##### There's no free surface with a rigid lid
#####

@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::Nothing) = zero(grid)
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::Nothing) = zero(grid)

materialize_free_surface(free_surface::Nothing, velocities, grid) = nothing
