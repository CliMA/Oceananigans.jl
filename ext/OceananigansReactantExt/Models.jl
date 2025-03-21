module Models

import Oceananigans.Models: setup_update_state!, inititalize!
import Oceananigans.Models.HydrostaticFreeSurfaceModels: set_barotropic_velocities!

using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces:
    compute_barotropic_velocities!

using ..Grids: ReactantGrid
using ..TimeSteppers: ReactantModel

setup_update_state!(::ReactantModel) = nothing
set_barotropic_velocities!(fs, ::ReactantGrid, velocities) = nothing

const ReactantHydrostaticFreeSurfaceModel = HydrostaticFreeSurfaceModel{<:Any, <:Any, <:ReactantState}

function initialize!(model::ReactantHydrostaticFreeSurfaceModel)

    if model.free_surface isa SplitExplicitFreeSurface
        U = model.free_surface.barotropic_velocities.U
        V = model.free_surface.barotropic_velocities.V
        η = model.free_surface.η
        u, v, w = model.velocities
        compute_barotropic_velocities!(U, V, grid, u, v, η)
    end

    return nothing
end


end # module
