using Oceananigans.DistributedComputations: DistributedField, AsynchronousDistributed, synchronize_communication!

import Oceananigans.DistributedComputations: synchronize_communication!

const DistributedSplitExplicit = SplitExplicitFreeSurface{<:Any, <:DistributedField}

wait_free_surface_communication!(free_surface, model, arch) = nothing

function wait_free_surface_communication!(free_surface::DistributedSplitExplicit, model, ::AsynchronousDistributed)

    barotropic_velocities = free_surface.barotropic_velocities

    synchronize_communication!(barotropic_velocities.U)
    synchronize_communication!(barotropic_velocities.V)

    Gᵁ = model.timestepper.Gⁿ.U
    Gⱽ = model.timestepper.Gⁿ.V

    synchronize_communication!(Gᵁ)
    synchronize_communication!(Gⱽ)

    Ω = free_surface.implicit_boundary_coefficients
    synchronize_communication!(Ω.U)
    synchronize_communication!(Ω.V)

    return nothing
end

function synchronize_communication!(free_surface::SplitExplicitFreeSurface)
    η    = free_surface.displacement
    U, V = free_surface.barotropic_velocities
    Ũ, Ṽ = free_surface.filtered_state.Ũ, free_surface.filtered_state.Ṽ

    synchronize_communication!(U)
    synchronize_communication!(V)
    synchronize_communication!(Ũ)
    synchronize_communication!(Ṽ)
    synchronize_communication!(η)

    return nothing
end
