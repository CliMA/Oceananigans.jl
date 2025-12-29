using .SplitExplicitFreeSurfaces: barotropic_split_explicit_corrector!

#####
##### Barotropic pressure correction for models with a free surface
#####

correct_barotropic_mode!(model::HydrostaticFreeSurfaceModel, Δt; kwargs...) =
    correct_barotropic_mode!(model, model.free_surface, Δt; kwargs...)

# Fallback for ExplicitFreeSurface and Nothing free surfaces 
correct_barotropic_mode!(model, ::Nothing, Δt; kwargs...) = nothing
correct_barotropic_mode!(model, ::ExplicitFreeSurface, Δt; kwargs...) = nothing

#####
##### Barotropic pressure correction for models with an Implicit free surface
#####

function correct_barotropic_mode!(model, ::ImplicitFreeSurface, Δt)

    launch!(model.architecture, model.grid, :xyz,
            _barotropic_pressure_correction!,
            model.velocities,
            model.grid,
            Δt,
            model.free_surface.gravitational_acceleration,
            model.free_surface.η)

    return nothing
end

function correct_barotropic_mode!(model, ::SplitExplicitFreeSurface, Δt)
    u, v, _ = model.velocities
    grid = model.grid
    barotropic_split_explicit_corrector!(u, v, model.free_surface, grid)

    return nothing
end

@kernel function _barotropic_pressure_correction!(U, grid, Δt, g, η)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U.u[i, j, k] -= g * Δt * ∂xᶠᶜᶠ(i, j, grid.Nz+1, grid, η)
        U.v[i, j, k] -= g * Δt * ∂yᶜᶠᶠ(i, j, grid.Nz+1, grid, η)
    end
end
