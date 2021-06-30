#####
##### There's no free surface with a rigid lid
#####

@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, free_surface::Nothing) = zero(eltype(grid))
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, free_surface::Nothing) = zero(eltype(grid))
