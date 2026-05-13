#####
##### There's no free surface with a rigid lid
#####

# Fallback: forward 4-arg call to 3-arg for free surfaces that don't need the clock
materialize_free_surface(fs, velocities, grid, clock) = materialize_free_surface(fs, velocities, grid)

@inline materialize_free_surface(::Nothing, velocities, grid) = nothing

@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::Nothing) = zero(grid)
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::Nothing) = zero(grid)
