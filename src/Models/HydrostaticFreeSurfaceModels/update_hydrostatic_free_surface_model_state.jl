using Oceananigans.Architectures
using Oceananigans.BoundaryConditions

using Oceananigans: UpdateStateCallsite
using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.Fields: replace_horizontal_vector_halos!
using Oceananigans.Grids: halo_size
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_field_xy!, inactive_node
using Oceananigans.Models: update_model_field_time_series!
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!, p_kernel_parameters
using Oceananigans.TurbulenceClosures: compute_diffusivities!

import Oceananigans.Models.NonhydrostaticModels: compute_auxiliaries!
import Oceananigans.TimeSteppers: update_state!

compute_auxiliary_fields!(auxiliary_fields) = Tuple(compute!(a) for a in auxiliary_fields)

# Note: see single_column_model_mode.jl for a "reduced" version of update_state! for
# single column models.

"""
    update_state!(model::HydrostaticFreeSurfaceModel, callbacks=[])

Update peripheral aspects of the model (auxiliary fields, halo regions, diffusivities,
hydrostatic pressure) to the current model state. If `callbacks` are provided (in an array),
they are called in the end.
"""
update_state!(model::HydrostaticFreeSurfaceModel, callbacks=[]; compute_tendencies = true) =
    update_state!(model, model.grid, callbacks; compute_tendencies)

operation_corner_points = "average" # Choose operation_corner_points to be "average", "CCW", or "CW".

function fill_velocity_halos!(velocities)
    u, v, _ = velocities
    grid = u.grid
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    for passes in 1:3
        fill_halo_regions!(u)
        fill_halo_regions!(v)
        @apply_regionally replace_horizontal_vector_halos!((; u, v, w = nothing), grid)
    end

    for region in [1, 3, 5]

        region_south = mod(region + 4, 6) + 1
        region_east = region + 1
        region_north = mod(region + 2, 6)
        region_west = mod(region + 4, 6)

        # Northwest corner region
        for k in -Hz+1:Nz+Hz
            # Local y direction
            # (a) Proceed from [1, Ny+1] to [1, Ny+Hy].
            # (b) Shift left by one index in the first dimension to proceed from [0, Ny+1] to [0, Ny+Hy].
            u[region][0, Ny+1:Ny+Hy, k] .= reverse(-u[region_west][2, Ny-Hy+1:Ny, k]')
            v[region][0, Ny+1, k] = -u[region][1, Ny, k]
            v[region][0, Ny+2:Ny+Hy, k] .= reverse(-v[region_west][1, Ny-Hy+2:Ny, k]')
            # Local x direction
            # (a) Proceed from [1-Hx, Ny] to [0, Ny].
            # (b) Shift up by one index in the second dimension to proceed from [1-Hx, Ny+1] to [0, Ny+1].
            u[region][1-Hx:0, Ny+1, k] .= reverse(-u[region_north][2:Hx+1, Ny, k])
            v[region][1-Hx:0, Ny+1, k] .= -u[region_west][1, Ny-Hx+1:Ny, k]
            # Corner point operation
            u_CCW = -u[region_west][2, Ny, k]
            u_CW = -u[region_north][2, Ny, k]
            u[region][0, Ny+1, k] = operation_corner_points == "average" ? 0.5 * (u_CCW + u_CW) :
                                    operation_corner_points == "CCW" ? u_CCW :
                                    operation_corner_points == "CW" ? u_CW : nothing
            v_CCW = -u[region][1, Ny, k] 
            v_CW = -u[region_west][1, Ny, k]
            v[region][0, Ny+1, k] = operation_corner_points == "average" ? 0.5 * (v_CCW + v_CW) :
                                    operation_corner_points == "CCW" ? v_CCW :
                                    operation_corner_points == "CW" ? v_CW : nothing
        end

        # Northeast corner region
        for k in -Hz+1:Nz+Hz
            # Local y direction
            # (a) Proceed from [Nx, Ny+1] to [Nx, Ny+Hy].
            # (b) Shift right by one index in the first dimension to proceed from [Nx+1, Ny+1] to [Nx+1, Ny+Hy].
            u[region][Nx+1, Ny+1:Ny+Hy, k] .= -v[region_north][1:Hy, 1, k]'
            v[region][Nx+1, Ny+1:Ny+Hy, k] .= u[region_east][1:Hy, Ny, k]'
            # Local x direction
            # (a) Proceed from [Nx+1, Ny] to [Nx+Hx, Ny].
            # (b) Shift up by one index in the second dimension to proceed from [Nx+1, Ny+1] to [Nx+Hx, Ny+1].
            u[region][Nx+1:Nx+Hx, Ny+1, k] .= u[region_north][1:Hx, 1, k]
            v[region][Nx+1:Nx+Hx, Ny+1, k] .= v[region_north][1:Hx, 1, k]
            # Corner point operation
            u_CCW = u[region_north][1, 1, k]
            u_CW = -v[region_north][1, 1, k]
            u[region][Nx+1, Ny+1, k] = operation_corner_points == "average" ? 0.5 * (u_CCW + u_CW) :
                                       operation_corner_points == "CCW" ? u_CCW :
                                       operation_corner_points == "CW" ? u_CW : nothing
            v_CCW = v[region_north][1, 1, k]
            v_CW = u[region_east][1, Ny, k]
            v[region][Nx+1, Ny+1, k] = operation_corner_points == "average" ? 0.5 * (v_CCW + v_CW) :
                                       operation_corner_points == "CCW" ? v_CCW :
                                       operation_corner_points == "CW" ? v_CW : nothing
        end

        # Southwest corner region
        for k in -Hz+1:Nz+Hz
            # Local y direction
            # (a) Proceed from [1, 1-Hy] to [1, 0].
            # (b) Shift left by one index in the first dimension to proceed from [0, 1-Hy] to [0, 0].
            u[region][0, 1-Hy:0, k] .= u[region_west][Nx, Ny-Hy+1:Ny, k]'
            v[region][0, 1-Hy:0, k] .= v[region_west][Nx, Ny-Hy+1:Ny, k]'
            # Local x direction
            # (a) Proceed from [1-Hx, 1] to [0, 1].
            # (b) Shift down by one index in the second dimension to proceed from [1-Hx, 0] to [0, 0].
            u[region][1-Hx:0, 0, k] .= v[region_south][1, Ny-Hx+1:Ny, k]
            v[region][1-Hx:0, 0, k] .= -u[region_south][2, Ny-Hx+1:Ny, k]
            # Corner point operation
            u_CCW = v[region_south][1, Ny, k]
            u_CW = u[region_west][Nx, Ny, k]
            u[region][0, 0, k] = operation_corner_points == "average" ? 0.5 * (u_CCW + u_CW) :
                                 operation_corner_points == "CCW" ? u_CCW :
                                 operation_corner_points == "CW" ? u_CW : nothing
            v_CCW = -u[region_south][2, Ny, k]
            v_CW = v[region_west][Nx, Ny, k]
            v[region][0, 0, k] = operation_corner_points == "average" ? 0.5 * (v_CCW + v_CW) :
                                 operation_corner_points == "CCW" ? v_CCW :
                                 operation_corner_points == "CW" ? v_CW : nothing
        end

        # Southeast corner region
        for k in -Hz+1:Nz+Hz
            # Local y direction
            # (a) Proceed from [Nx, 1-Hy] to [Nx, 0].
            # (b) Shift right by one index in the first dimension to proceed from [Nx+1, 1-Hy] to [Nx+1, 0].
            u[region][Nx+1, 1-Hy:0, k] .= reverse(v[region_east][1:Hy, 1, k]')
            v[region][Nx+1, 1-Hy:0, k] .= reverse(-u[region_east][2:Hy+1, 1, k]')
            # Local x direction
            # (a) Proceed from [Nx+1, 1] to [Nx+Hx, 1].
            # (b) Shift down by one index in the second dimension to proceed from [Nx+1, 0] to [Nx+Hx, 0].
            u[region][Nx+1, 0, k] = -v[region][Nx, 1, k]
            u[region][Nx+2:Nx+Hx, 0, k] .= reverse(-v[region_south][Nx, Ny-Hx+2:Ny, k])
            v[region][Nx+1:Nx+Hx, 0, k] .= u[region_south][Nx, Ny-Hx+1:Ny, k]
            # Corner point operation
            u_CCW = v[region_east][1, 1, k]
            u_CW = -v[region][Nx, 1, k]
            u[region][Nx+1, 0, k] = operation_corner_points == "average" ? 0.5 * (u_CCW + u_CW) :
                                    operation_corner_points == "CCW" ? u_CCW :
                                    operation_corner_points == "CW" ? u_CW : nothing
            v_CCW = -u[region_east][2, 1, k]
            v_CW = u[region_south][Nx, Ny, k]
            v[region][Nx+1, 0, k] = operation_corner_points == "average" ? 0.5 * (v_CCW + v_CW) :
                                    operation_corner_points == "CCW" ? v_CCW :
                                    operation_corner_points == "CW" ? v_CW : nothing
        end
    end
    
    for region in [2, 4, 6]
        region_south = mod(region + 3, 6) + 1
        region_east = mod(region, 6) + 2
        region_north = mod(region, 6) + 1
        region_west = region - 1

        # Northwest corner region
        for k in -Hz+1:Nz+Hz
            # Local y direction
            # (a) Proceed from [1, Ny+1] to [1, Ny+Hy].
            # (b) Shift left by one index in the first dimension to proceed from [0, Ny+1] to [0, Ny+Hy].
            u[region][0, Ny+1:Ny+Hy, k] .= reverse(v[region_west][Nx-Hy+1:Nx, Ny, k]')
            v[region][0, Ny+1, k] = -u[region][1, Ny, k]
            v[region][0, Ny+2:Ny+Hy, k] .= reverse(-u[region_west][Nx-Hy+2:Nx, Ny, k]')
            # Local x direction
            # (a) Proceed from [1-Hx, Ny] to [0, Ny].
            # (b) Shift up by one index in the second dimension to proceed from [1-Hx, Ny+1] to [0, Ny+1].
            u[region][1-Hx:0, Ny+1, k] .= reverse(-v[region_north][1, 2:Hx+1, k])
            v[region][1-Hx:0, Ny+1, k] .= reverse(u[region_north][1, 1:Hx, k])
            # Corner point operation
            u_CCW = v[region_west][Nx, Ny, k]
            u_CW = -v[region_north][1, 2, k]
            u[region][0, Ny+1, k] = operation_corner_points == "average" ? 0.5 * (u_CCW + u_CW) :
                                    operation_corner_points == "CCW" ? u_CCW :
                                    operation_corner_points == "CW" ? u_CW : nothing
            v_CCW = -u[region][1, Ny, k]
            v_CW = u[region_north][1, 1, k]
            v[region][0, Ny+1, k] = operation_corner_points == "average" ? 0.5 * (v_CCW + v_CW) :
                                    operation_corner_points == "CCW" ? v_CCW :
                                    operation_corner_points == "CW" ? v_CW : nothing    
        end

        # Northeast corner region
        for k in -Hz+1:Nz+Hz
            # Local y direction
            # (a) Proceed from [Nx, Ny+1] to [Nx, Ny+Hy].
            # (b) Shift right by one index in the first dimension to proceed from [Nx+1, Ny+1] to [Nx+1, Ny+Hy].
            u[region][Nx+1, Ny+1:Ny+Hy, k] .= u[region_east][1, 1:Hy, k]'
            v[region][Nx+1, Ny+1:Ny+Hy, k] .= v[region_east][1, 1:Hy, k]'
            # Local x direction
            # (a) Proceed from [Nx+1, Ny] to [Nx+Hx, Ny].
            # (b) Shift up by one index in the second dimension to proceed from [Nx+1, Ny+1] to [Nx+Hx, Ny+1].
            u[region][Nx+1:Nx+Hx, Ny+1, k] .= v[region_north][Nx, 1:Hx, k]
            v[region][Nx+1:Nx+Hx, Ny+1, k] .= -u[region_east][1, 1:Hx, k]
            # Corner point operation
            u_CCW = v[region_north][Nx, 1, k]
            u_CW = u[region_east][1, 1, k]
            u[region][Nx+1, Ny+1, k] = operation_corner_points == "average" ? 0.5 * (u_CCW + u_CW) :
                                       operation_corner_points == "CCW" ? u_CCW :
                                       operation_corner_points == "CW" ? u_CW : nothing
            v_CCW = -u[region_east][1, 1, k]
            v_CW = v[region_east][1, 1, k]
            v[region][Nx+1, Ny+1, k] = operation_corner_points == "average" ? 0.5 * (v_CCW + v_CW) :
                                       operation_corner_points == "CCW" ? v_CCW :
                                       operation_corner_points == "CW" ? v_CW : nothing
        end
        
        # Southwest corner region
        for k in -Hz+1:Nz+Hz
            # Local y direction
            # (a) Proceed from [1, 1-Hy] to [1, 0].
            # (b) Shift left by one index in the first dimension to proceed from [0, 1-Hy] to [0, 0].
            u[region][0, 1-Hy:0, k] .= -v[region_west][Nx-Hy+1:Nx, 2, k]'
            v[region][0, 1-Hy:0, k] .= u[region_west][Nx-Hy+1:Nx, 1, k]'
            # Local x direction
            # (a) Proceed from [1-Hx, 1] to [0, 1].
            # (b) Shift down by one index in the second dimension to proceed from [1-Hx, 0] to [0, 0].
            u[region][1-Hx:0, 0, k] .= u[region_south][Nx-Hx+1:Nx, Ny, k]
            v[region][1-Hx:0, 0, k] .= v[region_south][Nx-Hx+1:Nx, Ny, k]
            # Corner point operation
            u_CCW = u[region_south][Nx, Ny, k]
            u_CW = -v[region_west][Nx, 2, k]
            u[region][0, 0, k] = operation_corner_points == "average" ? 0.5 * (u_CCW + u_CW) :
                                 operation_corner_points == "CCW" ? u_CCW :
                                 operation_corner_points == "CW" ? u_CW : nothing
            v_CCW = v[region_south][Nx, Ny, k]
            v_CW = u[region_west][Nx, 1, k]
            v[region][0, 0, k] = operation_corner_points == "average" ? 0.5 * (v_CCW + v_CW) :
                                 operation_corner_points == "CCW" ? v_CCW :
                                 operation_corner_points == "CW" ? v_CW : nothing
        end
        
        # Southeast corner region
        for k in -Hz+1:Nz+Hz
            # Local y direction
            # (a) Proceed from [Nx, 1-Hy] to [Nx, 0].
            # (b) Shift right by one index in the first dimension to proceed from [Nx+1, 1-Hy] to [Nx+1, 0].
            u[region][Nx+1, 1-Hy:0, k] .= -v[region_south][Nx-Hy+1:Nx, 1, k]'
            v[region][Nx+1, 1-Hy:0, k] .= reverse(-v[region_east][Nx, 2:Hy+1, k]')
            # Local x direction
            # (a) Proceed from [Nx+1, 1] to [Nx+Hx, 1].
            # (b) Shift down by one index in the second dimension to proceed from [Nx+1, 0] to [Nx+Hx, 0].
            u[region][Nx+1, 0, k] = -v[region][Nx, 1, k]
            u[region][Nx+2:Nx+Hx, 0, k] .= reverse(-u[region_south][Nx-Hx+2:Nx, 1, k])
            v[region][Nx+1:Nx+Hx, 0, k] .= reverse(-v[region_south][Nx-Hx+1:Nx, 2, k])
            # Corner point operation
            u_CCW = -v[region_south][Nx, 1, k]
            u_CW = -v[region][Nx, 1, k]
            u[region][Nx+1, 0, k] = operation_corner_points == "average" ? 0.5 * (u_CCW + u_CW) :
                                    operation_corner_points == "CCW" ? u_CCW :
                                    operation_corner_points == "CW" ? u_CW : nothing
            v_CCW = -v[region_east][Nx, 2, k]
            v_CW = -v[region_south][Nx, 2, k]
            v[region][Nx+1, 0, k] = operation_corner_points == "average" ? 0.5 * (v_CCW + v_CW) :
                                    operation_corner_points == "CCW" ? v_CCW :
                                    operation_corner_points == "CW" ? v_CW : nothing
        end        
    end

    return nothing
end

function update_state!(model::HydrostaticFreeSurfaceModel, grid, callbacks; compute_tendencies = true)

    @apply_regionally mask_immersed_model_fields!(model, grid)

    # Update possible FieldTimeSeries used in the model
    @apply_regionally update_model_field_time_series!(model, model.clock)

    fill_halo_regions!(prognostic_fields(model), model.clock, fields(model); async = true)

    fill_velocity_halos!(model.velocities)
    # second_pass_of_fill_halo_regions!(grid, model.velocities, model.clock, fields(model))
    
    # @apply_regionally replace_horizontal_vector_halos!(model.velocities, model.grid)
    @apply_regionally compute_auxiliaries!(model)

    fill_halo_regions!(model.diffusivity_fields; only_local_halos = true)

    [callback(model) for callback in callbacks if callback.callsite isa UpdateStateCallsite]
    
    update_biogeochemical_state!(model.biogeochemistry, model)

    compute_tendencies &&
        @apply_regionally compute_tendencies!(model, callbacks)

    return nothing
end

# Mask immersed fields
function mask_immersed_model_fields!(model, grid)
    η = displacement(model.free_surface)
    fields_to_mask = merge(model.auxiliary_fields, prognostic_fields(model))

    foreach(fields_to_mask) do field
        if field !== η
            mask_immersed_field!(field)
        end
    end
    mask_immersed_field_xy!(η, k=size(grid, 3)+1, mask = inactive_node)

    return nothing
end

function compute_auxiliaries!(model::HydrostaticFreeSurfaceModel; w_parameters = tuple(w_kernel_parameters(model.grid)),
                                                                  p_parameters = tuple(p_kernel_parameters(model.grid)),
                                                                  κ_parameters = tuple(:xyz))

    grid = model.grid
    closure = model.closure
    diffusivity = model.diffusivity_fields

    for (wpar, ppar, κpar) in zip(w_parameters, p_parameters, κ_parameters)
        compute_w_from_continuity!(model; parameters = wpar)

        compute_diffusivities!(diffusivity, closure, model; parameters = κpar)

        update_hydrostatic_pressure!(model.pressure.pHY′, architecture(grid),
                                     grid, model.buoyancy, model.tracers;
                                     parameters = ppar)
    end

    return nothing
end

# TO DELETE!!!!! (We aim to do single pass)
second_pass_of_fill_halo_regions!(grid, velocities, args...) = nothing
