import Oceananigans.TimeSteppers: calculate_pressure_correction!, pressure_correct_velocities!

calculate_pressure_correction!(::HydrostaticFreeSurfaceModel, Δt) = nothing

#####
##### Barotropic pressure correction for models with a free surface
#####

const HFSM = HydrostaticFreeSurfaceModel
const ExplicitFreeSurfaceHFSM      = HFSM{<:Any, <:Any, <:Any, <:ExplicitFreeSurface}
const ImplicitFreeSurfaceHFSM      = HFSM{<:Any, <:Any, <:Any, <:ImplicitFreeSurface}
const SplitExplicitFreeSurfaceHFSM = HFSM{<:Any, <:Any, <:Any, <:SplitExplicitFreeSurface}

pressure_correct_velocities!(model::ExplicitFreeSurfaceHFSM, Δt; kwargs...) = nothing

#####
##### Barotropic pressure correction for models with a free surface
#####

function pressure_correct_velocities!(model::ImplicitFreeSurfaceHFSM, Δt)

    launch!(model.architecture, model.grid, :xyz,
            _barotropic_pressure_correction,
            model.velocities,
            model.grid,
            Δt,
            model.free_surface.gravitational_acceleration,
            model.free_surface.η)

    return nothing
end

calculate_free_surface_tendency!(grid, model::ImplicitFreeSurfaceHFSM     ) = nothing
calculate_free_surface_tendency!(grid, model::SplitExplicitFreeSurfaceHFSM) = nothing

function pressure_correct_velocities!(model::SplitExplicitFreeSurfaceHFSM, Δt)
    u, v, _ = model.velocities
    grid = model.grid 
    barotropic_split_explicit_corrector!(u, v, model.free_surface, grid)

    return nothing
end

@kernel function _barotropic_pressure_correction(U, grid, Δt, g, η)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U.u[i, j, k] -= g * Δt * ∂xᶠᶜᶠ(i, j, grid.Nz+1, grid, η)
        U.v[i, j, k] -= g * Δt * ∂yᶜᶠᶠ(i, j, grid.Nz+1, grid, η)
    end
end
