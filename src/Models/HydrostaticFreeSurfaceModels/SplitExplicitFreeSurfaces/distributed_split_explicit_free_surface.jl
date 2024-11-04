using Oceananigans.Grids: with_halo
using Oceananigans.AbstractOperations: GridMetricOperation, Δz
using Oceananigans.DistributedComputations: DistributedGrid, DistributedField
using Oceananigans.DistributedComputations: SynchronizedDistributed, synchronize_communication!

# Internal function for HydrostaticFreeSurfaceModel
function materialize_free_surface(free_surface::SplitExplicitFreeSurface, velocities, grid::DistributedGrid)

        old_halos  = halo_size(grid)
        Nsubsteps  = length(free_surface.substepping.averaging_weights)

        extended_halos = distributed_split_explicit_halos(old_halos, Nsubsteps+1, grid)         
        extended_grid  = with_halo(extended_halos, grid)

        η = free_surface_displacement_field(velocities, free_surface, extended_grid)
        η̅ = free_surface_displacement_field(velocities, free_surface, extended_grid)
    
        u_baroclinic = velocities.u
        v_baroclinic = velocities.v
    
        u_bc = barotropic_bc(u_baroclinic)
        v_bc = barotropic_bc(v_baroclinic)
    
        U = Field{Center, Center, Nothing}(extended_grid, boundary_conditions = u_bc)
        V = Field{Center, Center, Nothing}(extended_grid, boundary_conditions = v_bc)
    
        U̅ = Field{Center, Center, Nothing}(extended_grid, boundary_conditions = u_bc)
        V̅ = Field{Center, Center, Nothing}(extended_grid, boundary_conditions = v_bc)
    
        filtered_state = (η = η̅, U = U̅, V = V̅)
        barotropic_velocities = (U = U, V = V)
    
        gravitational_acceleration = convert(eltype(extended_grid), free_surface.gravitational_acceleration)
        timestepper = materialize_timestepper(free_surface.timestepper, extended_grid, free_surface, velocities, u_bc, v_bc)
    
        # In a non-parallel grid we calculate only the interior
        kernel_size    = augmented_kernel_size(extended_grid)
        kernel_offsets = augmented_kernel_offsets(extended_grid)

        kernel_parameters = KernelParameters(kernel_size, kernel_offsets)

        return SplitExplicitFreeSurface(η,
                                        barotropic_velocities,
                                        filtered_state,
                                        gravitational_acceleration,
                                        kernel_parameters,
                                        free_surface.substepping,
                                        timestepper)
end

@inline function distributed_split_explicit_halos(old_halos, step_halo, grid::DistributedGrid)

    Rx, Ry, _ = architecture(grid).ranks

    Ax = Rx == 1 ? old_halos[1] : max(step_halo, old_halos[1])
    Ay = Ry == 1 ? old_halos[2] : max(step_halo, old_halos[2])

    return (Ax, Ay, old_halos[3])
end

@inline function augmented_kernel_size(grid::DistributedGrid)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    Tx, Ty, _ = topology(grid)

    Rx, Ry, _ = architecture(grid).ranks

    Ax = Rx == 1 ? Nx : (Tx == RightConnected || Tx == LeftConnected ? Nx + Hx - 1 : Nx + 2Hx - 2)
    Ay = Ry == 1 ? Ny : (Ty == RightConnected || Ty == LeftConnected ? Ny + Hy - 1 : Ny + 2Hy - 2)

    return (Ax, Ay)
end
   
@inline function augmented_kernel_offsets(grid::DistributedGrid)
    Hx, Hy, _ = halo_size(grid)
    Tx, Ty, _ = topology(grid)

    Rx, Ry, _ = architecture(grid).ranks

    Ax = Rx == 1 || Tx == RightConnected ? 0 : - Hx + 1
    Ay = Ry == 1 || Ty == RightConnected ? 0 : - Hy + 1

    return (Ax, Ay)
end

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
