using Oceananigans.Models: AbstractModel
using Oceananigans.Models.HydrostaticFreeSurfaceModels: AbstractFreeSurface
using Oceananigans.TimeSteppers: AbstractTimeStepper
using Oceananigans.Models: PrescribedVelocityFields
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans: prognostic_fields, fields
using Oceananigans.Advection: AbstractAdvectionScheme

import Oceananigans.Advection: WENO5

import Oceananigans.Models.HydrostaticFreeSurfaceModels:
                        HydrostaticFreeSurfaceModel,
                        build_implicit_step_solver,
                        validate_tracer_advection

import Oceananigans.TurbulenceClosures: implicit_diffusion_solver

const MultiRegionModel = HydrostaticFreeSurfaceModel{<:Any, <:Any, <:AbstractArchitecture, <:Any, <:MultiRegionGrid}

# Bottleneck is getregion!!! (there are type issues with FieldBoundaryConditions and with propertynames)
@inline @inbounds getregion(mr::AbstractModel, i)            = getname(mr)(Tuple(getregion(getproperty(mr, propertynames(mr)[idx]), i) for idx in 1:length(propertynames(mr)))...)
@inline @inbounds getregion(ts::AbstractTimeStepper, i)      = getname(ts)(Tuple(getregion(getproperty(ts, propertynames(ts)[idx]), i) for idx in 1:length(propertynames(ts)))...)
@inline @inbounds getregion(fs::AbstractFreeSurface, i)      = getname(fs)(Tuple(getregion(getproperty(fs, propertynames(fs)[idx]), i) for idx in 1:length(propertynames(fs)))...)
@inline @inbounds getregion(pv::PrescribedVelocityFields, i) = getname(pv)(Tuple(getregion(getproperty(pv, propertynames(pv)[idx]), i) for idx in 1:length(propertynames(pv)))...)

@inline @inbounds getregion(fs::ExplicitFreeSurface, i) =
                    ExplicitFreeSurface(getregion(fs.η, i), fs.gravitational_acceleration)

isregional(pv::PrescribedVelocityFields) = isregional(pv.u) | isregional(pv.v) | isregional(pv.w)
devices(pv::PrescribedVelocityFields)    = devices(pv[findfirst(isregional, (pv.u, pv.v, pv.w))])

validate_tracer_advection(tracer_advection::MultiRegionObject, grid::MultiRegionGrid) = tracer_advection, NamedTuple()

isregional(mrm::MultiRegionModel)        = true
devices(mrm::MultiRegionModel)           = devices(mrm.grid)
getdevice(mrm::MultiRegionModel, i)      = getdevice(mrm.grid, i)
switch_region!(mrm::MultiRegionModel, i) = switch_region!(mrm.grid, i)

implicit_diffusion_solver(time_discretization::VerticallyImplicitTimeDiscretization, mrg::MultiRegionGrid) =
      construct_regionally(implicit_diffusion_solver, time_discretization, mrg)
