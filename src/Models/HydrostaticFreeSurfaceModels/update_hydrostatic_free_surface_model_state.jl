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

operation_corner_points = "default" # Choose operation_corner_points to be "default", "average", "CCW", or "CW".

function fill_velocity_halos!(velocities)
    u, v, _ = velocities
    grid = u.grid
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)
    plmn = -1
#- will not work if (Nx,Hx) and (Ny,Hy) are not equal
    Nc = Nx ; Hc = Hx

#=
    for passes in 1:3
        fill_halo_regions!(u)
        fill_halo_regions!(v)
        @apply_regionally replace_horizontal_vector_halos!((; u, v, w = nothing), grid)
    end
=#

    #-- first pass: only take interior-point value:
    for region in 1:6

      if mod(region,2) == 1
        #- odd face number (1,3,5):
        region_E = mod(region + 0, 6) + 1
        region_N = mod(region + 1, 6) + 1
        region_W = mod(region + 3, 6) + 1
        region_S = mod(region + 4, 6) + 1
        for k in -Hz+1:Nz+Hz
        #- E + W Halo for U:
            u[region][Nc+1:Nc+Hc, 1:Nc, k] .=     u[region_E][1:Hc, 1:Nc, k]
            u[region][1-Hc:0, 1:Nc, k] .= reverse(v[region_W][1:Nc, Nc+1-Hc:Nc, k],dims=1)'
        #- N + S Halo for V:
            v[region][1:Nc, Nc+1:Nc+Hc, k] .= reverse(u[region_N][1:Hc, 1:Nc, k],dims=2)'
            v[region][1:Nc, 1-Hc:0, k] .=             v[region_S][1:Nc, Nc+1-Hc:Nc, k]
        end
      else
        #- even face number (2,4,6):
        region_E = mod(region + 1, 6) + 1
        region_N = mod(region + 0, 6) + 1
        region_W = mod(region + 4, 6) + 1
        region_S = mod(region + 3, 6) + 1
        for k in -Hz+1:Nz+Hz
        #- E + W Halo for U:
            u[region][Nc+1:Nc+Hc, 1:Nc, k] .= reverse(v[region_E][1:Nc, 1:Hc, k],dims=1)'
            u[region][1-Hc:0, 1:Nc, k] .=             u[region_W][Nc+1-Hc:Nc,  1:Nc, k]
        #- N + S Halo for V:
            v[region][1:Nc, Nc+1:Nc+Hc, k] .=     v[region_N][1:Nc, 1:Hc, k]
            v[region][1:Nc, 1-Hc:0, k] .= reverse(u[region_S][Nc+1-Hc:Nc, 1:Nc, k],dims=2)'
        end
      end

    end

    #-- Second pass: fill the remaining halo:
    # iMn = 1 ; iMx = Nc+1     #- filling over this range is neccessary
    iMn = 2-Hc ; iMx = Nc+Hc   #- this will also fill corner halos with useless values
    for region in 1:6

      if mod(region,2) == 1
        #- odd face number (1,3,5):
        region_E = mod(region + 0, 6) + 1
        region_N = mod(region + 1, 6) + 1
        region_W = mod(region + 3, 6) + 1
        region_S = mod(region + 4, 6) + 1
        for k in -Hz+1:Nz+Hz
        #- N + S Halo for U:
            u[region][iMn:iMx, Nc+1:Nc+Hc, k] .= reverse(v[region_N][1:Hc, iMn:iMx, k],dims=2)'*plmn
            u[region][iMn:iMx, 1-Hc:0, k] .=             u[region_S][iMn:iMx, Nc+1-Hc:Nc, k]
        #- E + W Halo for V:
            v[region][Nc+1:Nc+Hc, iMn:iMx, k] .=     v[region_E][1:Hc, iMn:iMx, k]
            v[region][1-Hc:0, iMn:iMx, k] .= reverse(u[region_W][iMn:iMx, Nc+1-Hc:Nc, k],dims=1)'*plmn
        end
      else
        #- even face number (2,4,6):
        region_E = mod(region + 1, 6) + 1
        region_N = mod(region + 0, 6) + 1
        region_W = mod(region + 4, 6) + 1
        region_S = mod(region + 3, 6) + 1
        for k in -Hz+1:Nz+Hz
        #- N + S Halo for U:
            u[region][iMn:iMx, Nc+1:Nc+Hc, k] .=     u[region_N][iMn:iMx, 1:Hc, k]
            u[region][iMn:iMx, 1-Hc:0, k] .= reverse(v[region_S][Nc+1-Hc:Nc, iMn:iMx, k],dims=2)'*plmn
        #- E + W Halo for V:
            v[region][Nc+1:Nc+Hc, iMn:iMx, k] .= reverse(u[region_E][iMn:iMx, 1:Hc, k],dims=1)'*plmn
            v[region][1-Hc:0, iMn:iMx, k] .=             v[region_W][Nc+1-Hc:Nc, iMn:iMx, k]
        end
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
