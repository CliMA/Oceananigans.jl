using Oceananigans.TurbulenceClosures: calculate_diffusivities!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_reduced_field_xy!
using Oceananigans.Models: AbstractModel
using Oceananigans
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: AbstractFreeSurface, compute_w_from_continuity!, compute_auxiliary_fields!
using Oceananigans.TimeSteppers: AbstractTimeStepper, Clock
using Oceananigans.Models: PrescribedVelocityFields

using Oceananigans: prognostic_fields, fields

import Oceananigans.Models.HydrostaticFreeSurfaceModels:
                        HydrostaticFreeSurfaceModel,
                        validate_vertical_velocity_boundary_conditions

import Oceananigans.TimeSteppers: 
                        ab2_step!,
                        update_state!,
                        calculate_tendencies!,
                        store_tendencies!

using Oceananigans.Simulations

const MultiRegionModel      = HydrostaticFreeSurfaceModel{<:Any, <:Any, <:AbstractArchitecture, <:Any, <:MultiRegionGrid}
const MultiRegionSimulation = Simulation{<:MultiRegionModel}

function update_state!(mrm::MultiRegionModel, mrg::MultiRegionGrid)

    # No Masking for the moment: Remember to apply masking!!
    fill_halo_regions!(prognostic_fields(mrm), mrm.architecture, mrm.clock, fields(mrm))

    @apply_regionally begin
        update_state_actions!(mrm)
    end

    fill_halo_regions!(mrm.velocities.w, mrm.architecture, mrm.clock, fields(mrm))
    fill_halo_regions!(mrm.diffusivity_fields, mrm.architecture, mrm.clock, fields(mrm))
    fill_halo_regions!(mrm.pressure.pHY′, mrm.architecture)
    
    return nothing
end

function update_state_actions!(mrm) 
    compute_w_from_continuity!(mrm)
    calculate_diffusivities!(mrm.diffusivity_fields, mrm.closure, mrm)
    update_hydrostatic_pressure!(mrm.pressure.pHY′, mrm.architecture, mrm.grid, mrm.buoyancy, mrm.tracers)
end

# Bottleneck is getregion!!! (there are type issues with FieldBoundaryConditions and with propertynames)
getregion(mr::AbstractModel, i)            = getname(mr)(Tuple(getregion(getproperty(mr, propertynames(mr)[idx]), i) for idx in 1:length(propertynames(mr)))...)
getregion(ts::AbstractTimeStepper, i)      = getname(ts)(Tuple(getregion(getproperty(ts, propertynames(ts)[idx]), i) for idx in 1:length(propertynames(ts)))...)
getregion(fs::AbstractFreeSurface, i)      = getname(fs)(Tuple(getregion(getproperty(fs, propertynames(fs)[idx]), i) for idx in 1:length(propertynames(fs)))...)
getregion(pv::PrescribedVelocityFields, i) = getname(pv)(Tuple(getregion(getproperty(pv, propertynames(pv)[idx]), i) for idx in 1:length(propertynames(pv)))...)

getregion(c::Clock, i)                     = Clock(time = 0)

getregion(fs::ExplicitFreeSurface, i) =
     ExplicitFreeSurface(getregion(fs.η, i), fs.gravitational_acceleration)

getname(type) = typeof(type).name.wrapper

isregional(mrm::MultiRegionModel)        = true
devices(mrm::MultiRegionModel)           = devices(mrm.grid)
getdevice(mrm::MultiRegionModel, i)      = getdevice(mrm.grid, i)
switch_region!(mrm::MultiRegionModel, i) = switch_region!(mrm.grid, i)

validate_vertical_velocity_boundary_conditions(w::MultiRegionField) = apply_regionally!(validate_vertical_velocity_boundary_conditions, w)