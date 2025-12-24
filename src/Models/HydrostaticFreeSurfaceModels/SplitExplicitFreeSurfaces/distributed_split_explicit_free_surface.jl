using Oceananigans.DistributedComputations: DistributedField
using Oceananigans.DistributedComputations: AsynchronousDistributed, synchronize_communication!

import Oceananigans.DistributedComputations: synchronize_communication!

const DistributedSplitExplicit = SplitExplicitFreeSurface{<:DistributedField}

wait_free_surface_communication!(free_surface, model, arch) = nothing

function wait_free_surface_communication!(free_surface::DistributedSplitExplicit, model, ::AsynchronousDistributed)

    barotropic_velocities = free_surface.barotropic_velocities

    for field in (barotropic_velocities.U, barotropic_velocities.V)
        synchronize_communication!(field)
    end

    Gᵁ = model.timestepper.Gⁿ.U
    Gⱽ = model.timestepper.Gⁿ.V

    for field in (Gᵁ, Gⱽ)
        synchronize_communication!(field)
    end

    return nothing
end

function synchronize_communication!(free_surface::SplitExplicitFreeSurface)
    η    = free_surface.η
    U, V = free_surface.barotropic_velocities
    Ũ, Ṽ = free_surface.filtered_state.Ũ, free_surface.filtered_state.Ṽ

    for field in (U, V, Ũ, Ṽ, η)
        synchronize_communication!(field)
    end
    
    return nothing
end
