using Oceananigans.TurbulenceClosures: calculate_diffusivities!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_reduced_field_xy!
using Oceananigans.Models
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_w_from_continuity!, displacement, compute_auxiliary_fields!
using Oceananigans.TimeSteppers: AbstractTimeStepper

using Oceananigans: prognostic_fields, fields

import Oceananigans.TimeSteppers: 
                        ab2_step!,
                        update_state!,
                        calculate_tendencies!,
                        store_tendencies!


const MultiRegionModel = HydrostaticFreeSurfaceModel{<:Any, <:Any, <:AbstractArchitecture, <:Any, <:MultiRegionGrid}

function update_state!(mrm::MultiRegionModel, mrg::MultiRegionGrid)

    # No Masking for the moment: Remember to apply masking!!
    fill_halo_regions!(prognostic_fields(mrm), mrm.architecture, mrm.clock, fields(mrm))
    
    apply_regionally!(compute_w_from_continuity!, mrm)
    
    fill_halo_regions!(mrm.velocities.w, mrm.architecture, mrm.clock, fields(mrm))
    
    apply_regionally!(compute_auxiliary_fields!, mrm.auxiliary_fields)

    # Calculate diffusivities
    apply_regionally!(calculate_diffusivities!, mrm.diffusivity_fields, mrm.closure, mrm)
    
    fill_halo_regions!(mrm.diffusivity_fields, mrm.architecture, mrm.clock, fields(mrm))
    
    apply_regionally!(update_hydrostatic_pressure!, mrm.pressure.pHY′, mrm.architecture, mrm.grid, mrm.buoyancy, mrm.tracers)
    
    fill_halo_regions!(mrm.pressure.pHY′, mrm.architecture)

    return nothing
end
 
ab2_step!(mrm::MultiRegionModel, Δt, χ)      = apply_regionally!(ab2_step!, mrm, Δt, χ)
calculate_tendencies!(mrm::MultiRegionModel) = apply_regionally!(calculate_tendencies!, mrm)
store_tendencies!(mrm::MultiRegionModel)     = apply_regionally!(store_tendencies!, mrm)

getregion(mrm::MultiRegionModel, i) = getname(mrm)([getregion(getproperty(mrm, name), i) for name in propertynames(mrm)]...)

# getregion(f::D, i) where D <: DataType    = getname(f)([getregion(getproperty(f, name), i) for name in propertynames(f)]...)
getregion(tstep::AbstractTimeStepper, i)  = getname(tstep)([getregion(getproperty(tstep, name), i) for name in propertynames(tstep)]...)
getregion(t::Tuple, i)                    = Tuple(getregion(elem, i) for elem in t)
getregion(nt::NamedTuple, i)              = NamedTuple{keys(nt)}(getregion(elem, i) for elem in nt)

getname(type) = typeof(type).name.wrapper

isregional(mrm::MultiRegionModel)        = true
devices(mrm::MultiRegionModel)           = devices(mrm.grid)
getdevice(mrm::MultiRegionModel, i)      = getdevice(mrm.grid, i)
switch_region!(mrm::MultiRegionModel, i) = switch_region!(mrm.grid, i)