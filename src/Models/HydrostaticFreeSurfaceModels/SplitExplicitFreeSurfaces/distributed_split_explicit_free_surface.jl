using Oceananigans.DistributedComputations: DistributedField
using Oceananigans.DistributedComputations: SynchronizedDistributed, synchronize_communication!

const DistributedSplitExplicit = SplitExplicitFreeSurface{<:DistributedField}

wait_free_surface_communication!(free_surface, model, arch) = nothing
wait_free_surface_communication!(::DistributedSplitExplicit, model, ::SynchronizedDistributed) = nothing
    
function wait_free_surface_communication!(free_surface::DistributedSplitExplicit, model, arch)
    
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
