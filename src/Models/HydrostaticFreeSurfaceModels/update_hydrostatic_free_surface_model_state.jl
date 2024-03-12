using Oceananigans.Architectures
using Oceananigans.BoundaryConditions

using Oceananigans: UpdateStateCallsite
using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Oceananigans.Grids: halo_size
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_field_xy!, inactive_node
using Oceananigans.Models: update_model_field_time_series!
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!, p_kernel_parameters
using Oceananigans.Fields: replace_horizontal_vector_halos!

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

function fill_paired_halo_regions!(fields, signed=true)

    field_1, field_2 = fields
    grid = field_1.grid

    if !(grid isa ConformalCubedSphereGrid)
        return
    end

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)
    signed ? plmn = -1 : plmn = 1
    #- will not work if (Nx,Hx) and (Ny,Hy) are not equal
    Nc = Nx ; Hc = Hx

    #-- one pass: only use interior-point values:
    for region in 1:6

        if mod(region,2) == 1
            #- odd face number (1,3,5):
            region_E = mod(region + 0, 6) + 1
            region_N = mod(region + 1, 6) + 1
            region_W = mod(region + 3, 6) + 1
            region_S = mod(region + 4, 6) + 1
            for k in -Hz+1:Nz+Hz
                #- E + W Halo for field_1:
                field_1[region][Nc+1:Nc+Hc, 1:Nc, k]   .=         field_1[region_E][1:Hc, 1:Nc, k]
                field_1[region][1-Hc:0, 1:Nc, k]       .= reverse(field_2[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)'
                #- N + S Halo for field_1:
                field_1[region][2:Nc+1, Nc+1:Nc+Hc, k] .= reverse(field_2[region_N][1:Hc, 1:Nc, k], dims=2)' * plmn
                field_1[region][1, Nc+1:Nc+Hc, k]       = reverse(field_1[region_W][1, Nc+1-Hc:Nc, k]) * plmn
                field_1[region][1:Nc, 1-Hc:0, k]       .=         field_1[region_S][1:Nc, Nc+1-Hc:Nc, k]
                field_1[region][Nc+1, 1-Hc:0, k]        = reverse(field_2[region_E][1:Hc, 1, k])
                #- E + W Halo for field_2:
                field_2[region][Nc+1:Nc+Hc, 1:Nc, k]   .=         field_2[region_E][1:Hc, 1:Nc, k]
                field_2[region][Nc+1:Nc+Hc, Nc+1, k]   .=         field_2[region_N][1:Hc, 1, k]
                field_2[region][1-Hc:0, 2:Nc+1, k]     .= reverse(field_1[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)' * plmn
                field_2[region][1-Hc:0, 1, k]          .=         field_1[region_S][1, Nc+1-Hc:Nc, k] * plmn
                #- N + S Halo for field_2:
                field_2[region][1:Nc, Nc+1:Nc+Hc, k]   .= reverse(field_1[region_N][1:Hc, 1:Nc, k], dims=2)'
                field_2[region][1:Nc, 1-Hc:0, k]       .=         field_2[region_S][1:Nc, Nc+1-Hc:Nc, k]
            end
        else
            #- even face number (2,4,6):
            region_E = mod(region + 1, 6) + 1
            region_N = mod(region + 0, 6) + 1
            region_W = mod(region + 4, 6) + 1
            region_S = mod(region + 3, 6) + 1
            for k in -Hz+1:Nz+Hz
                #- E + W Halo for field_1:
                field_1[region][Nc+1:Nc+Hc, 1:Nc, k]   .= reverse(field_2[region_E][1:Nc, 1:Hc, k], dims=1)'
                field_1[region][1-Hc:0, 1:Nc, k]       .=         field_1[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                #- N + S Halo for field_1:
                field_1[region][1:Nc, Nc+1:Nc+Hc, k]   .=         field_1[region_N][1:Nc, 1:Hc, k]
                field_1[region][Nc+1, Nc+1:Nc+Hc, k]    =         field_1[region_E][1, 1:Hc, k]
                field_1[region][2:Nc+1, 1-Hc:0, k]     .= reverse(field_2[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)' * plmn
                field_1[region][1, 1-Hc:0, k]           =         field_2[region_W][Nc+1-Hc:Nc, 1, k] * plmn
                #- E + W Halo for field_2:
                field_2[region][Nc+1:Nc+Hc, 2:Nc+1, k] .= reverse(field_1[region_E][1:Nc, 1:Hc, k], dims=1)' * plmn
                field_2[region][Nc+1:Nc+Hc, 1, k]      .= reverse(field_2[region_S][Nc+1-Hc:Nc, 1, k]) * plmn
                field_2[region][1-Hc:0, 1:Nc, k]       .=         field_2[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                field_2[region][1-Hc:0, Nc+1, k]       .= reverse(field_1[region_N][1, 1:Hc, k])
                #- N + S Halo for field_2:
                field_2[region][1:Nc, Nc+1:Nc+Hc, k]   .=         field_2[region_N][1:Nc, 1:Hc, k]
                field_2[region][1:Nc, 1-Hc:0, k]       .= reverse(field_1[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)'
            end
        end

    end

    #-- Add one valid field_1, field_2 value next to the corner, that allows
    #   to compute vorticity on a wider stencil (e.g., vort3(0,1) & (1,0)).
    for region in 1:6
        # println("region=",region,", size(field_1)=",size(field_1[region][:,:,1]),
        #                          ", size(field_2)=",size(field_2[region][:,:,1]))
        for k in -Hz+1:Nz+Hz
            #- SW corner:
            field_1[region][1-Hc:0, 0, k] .= field_2[region][1, 1-Hc:0, k]
            field_2[region][0, 1-Hc:0, k] .= field_1[region][1-Hc:0, 1, k]'
        end
        if Hc > 1
            for k in -Hz+1:Nz+Hz
                #- NW corner:
                field_1[region][2-Hc:0, Nc+1,  k]    .= reverse(field_2[region][1, Nc+2:Nc+Hc, k]) * plmn
                field_2[region][0, Nc+2:Nc+Hc, k]    .= reverse(field_1[region][2-Hc:0, Nc, k])' * plmn
                #- SE corner:
                field_1[region][Nc+2:Nc+Hc, 0, k]    .= reverse(field_2[region][Nc, 2-Hc:0, k]) * plmn
                field_2[region][Nc+1, 2-Hc:0,  k]    .= reverse(field_1[region][Nc+2:Nc+Hc, 1, k])' * plmn
                #- NE corner:
                field_1[region][Nc+2:Nc+Hc, Nc+1, k] .= field_2[region][Nc, Nc+2:Nc+Hc, k]
                field_2[region][Nc+1, Nc+2:Nc+Hc, k] .= field_1[region][Nc+2:Nc+Hc, Nc, k]'
            end
        end
    end

    return nothing
end

function fill_paired_faceface_halo_regions!(fields, signed=true)

    field_1, field_2 = fields
    grid = field_1.grid

    if !(grid isa ConformalCubedSphereGrid)
        return
    end

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)
    signed ? plmn = -1 : plmn = 1
    #- will not work if (Nx,Hx) and (Ny,Hy) are not equal
    Nc = Nx ; Hc = Hx

    #-- one pass: only use interior-point values:
    for region in 1:6

        if mod(region,2) == 1
            #- odd face number (1,3,5):
            region_E = mod(region + 0, 6) + 1
            region_N = mod(region + 1, 6) + 1
            region_W = mod(region + 3, 6) + 1
            region_S = mod(region + 4, 6) + 1
            for k in -Hz+1:Nz+Hz
                #- E + W Halo for field_1:
                field_1[region][Nc+1:Nc+Hc, 1:Nc, k]   .=         field_1[region_E][1:Hc, 1:Nc, k]
                field_1[region][1-Hc:0, 2:Nc+1, k]     .= reverse(field_2[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)'
                field_1[region][1-Hc:0, 1, k]          .=         field_2[region_S][1, Nc+1-Hc:Nc, k]
                #- N + S Halo for field_1:
                field_1[region][2:Nc+1, Nc+1:Nc+Hc, k] .= reverse(field_2[region_N][1:Hc, 1:Nc, k], dims=2)' * plmn
                if Hc > 1
                    field_1[region][1, Nc+2:Nc+Hc, k]  .= reverse(field_1[region_W][1, Nc+2-Hc:Nc, k]) * plmn
                end
                field_1[region][1:Nc, 1-Hc:0, k]       .=         field_1[region_S][1:Nc, Nc+1-Hc:Nc, k]
                field_1[region][Nc+1, 1-Hc:0, k]        = reverse(field_2[region_E][2:Hc+1, 1, k])
                #- E + W Halo for field_2:
                field_2[region][Nc+1:Nc+Hc, 1:Nc, k]   .=         field_2[region_E][1:Hc, 1:Nc, k]
                field_2[region][Nc+1:Nc+Hc, Nc+1, k]   .=         field_2[region_N][1:Hc, 1, k]
                field_2[region][1-Hc:0, 2:Nc+1, k]     .= reverse(field_1[region_W][1:Nc, Nc+1-Hc:Nc, k], dims=1)' * plmn
                field_2[region][1-Hc:0, 1, k]          .=         field_1[region_S][1, Nc+1-Hc:Nc, k] * plmn
                #- N + S Halo for field_2:
                field_2[region][2:Nc, Nc+1:Nc+Hc, k]   .= reverse(field_1[region_N][1:Hc, 2:Nc, k], dims=2)'
                if Hc > 1
                    field_2[region][1, Nc+2:Nc+Hc, k]  .= reverse(field_2[region_W][1, Nc+2-Hc:Nc, k]) * plmn
                end
                field_2[region][1:Nc, 1-Hc:0, k]       .=         field_2[region_S][1:Nc, Nc+1-Hc:Nc, k]
            end
        else
            #- even face number (2,4,6):
            region_E = mod(region + 1, 6) + 1
            region_N = mod(region + 0, 6) + 1
            region_W = mod(region + 4, 6) + 1
            region_S = mod(region + 3, 6) + 1
            for k in -Hz+1:Nz+Hz
                #- E + W Halo for field_1:
                field_1[region][Nc+1:Nc+Hc, 2:Nc, k]   .= reverse(field_2[region_E][2:Nc, 1:Hc, k], dims=1)'
                if Hc > 1
                    field_1[region][Nc+2:Nc+Hc, 1, k]  .= reverse(field_1[region_S][Nc+2-Hc:Nc, 1, k]) * plmn
                end
                field_1[region][1-Hc:0, 1:Nc, k]       .=         field_1[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                #- N + S Halo for field_1:
                field_1[region][1:Nc, Nc+1:Nc+Hc, k]   .=         field_1[region_N][1:Nc, 1:Hc, k]
                field_1[region][Nc+1, Nc+1:Nc+Hc, k]    =         field_1[region_E][1, 1:Hc, k]
                field_1[region][2:Nc+1, 1-Hc:0, k]     .= reverse(field_2[region_S][Nc+1-Hc:Nc, 1:Nc, k], dims=2)' * plmn
                field_1[region][1, 1-Hc:0, k]           =         field_2[region_W][Nc+1-Hc:Nc, 1, k] * plmn
                #- E + W Halo for field_2:
                field_2[region][Nc+1:Nc+Hc, 2:Nc+1, k] .= reverse(field_1[region_E][1:Nc, 1:Hc, k], dims=1)' * plmn
                if Hc > 1
                    field_2[region][Nc+2:Nc+Hc, 1, k]  .= reverse(field_2[region_S][Nc+2-Hc:Nc, 1, k]) * plmn
                end
                field_2[region][1-Hc:0, 1:Nc, k]       .=         field_2[region_W][Nc+1-Hc:Nc, 1:Nc, k]
                field_2[region][1-Hc:0, Nc+1, k]       .= reverse(field_1[region_N][1, 2:Hc+1, k])
                #- N + S Halo for field_2:
                field_2[region][1:Nc, Nc+1:Nc+Hc, k]   .=         field_2[region_N][1:Nc, 1:Hc, k]
                field_2[region][2:Nc, 1-Hc:0, k]       .= reverse(field_1[region_S][Nc+1-Hc:Nc, 2:Nc, k], dims=2)'
                field_2[region][1, 1-Hc:0, k]           =         field_1[region_W][Nc+1-Hc:Nc, 1, k]
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
    @apply_regionally replace_horizontal_vector_halos!(model.velocities, model.grid)
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
