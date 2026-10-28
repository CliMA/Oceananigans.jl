#####
##### There's no free surface with a rigid lid
#####

# Only a PrescribedFreeSurface needs the clock
materialize_free_surface(free_surface, velocities, grid, bcs, clock) = materialize_free_surface(free_surface, velocities, grid, bcs)

@inline materialize_free_surface(::Nothing, velocities, grid, bcs) = nothing

@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::Nothing) = zero(grid)
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::Nothing) = zero(grid)
