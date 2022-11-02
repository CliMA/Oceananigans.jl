using Oceananigans.Models: AbstractModel
using Oceananigans.Advection: WENO
using Oceananigans.Models.HydrostaticFreeSurfaceModels: AbstractFreeSurface
using Oceananigans.TimeSteppers: AbstractTimeStepper, QuasiAdamsBashforth2TimeStepper, RungeKutta3TimeStepper
using Oceananigans.Models: PrescribedVelocityFields
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.Advection: AbstractAdvectionScheme

import Oceananigans.TimeSteppers: ab2_step!
import Oceananigans.Simulations: new_time_step
import Oceananigans.Diagnostics: accurate_cell_advection_timescale
import Oceananigans.Advection: WENO
import Oceananigans.Models.HydrostaticFreeSurfaceModels: build_implicit_step_solver, validate_tracer_advection
import Oceananigans.TurbulenceClosures: implicit_diffusion_solver

const MultiRegionModel = Union{HydrostaticFreeSurfaceModel{<:Any, <:Any, <:AbstractArchitecture, <:Any, <:MultiRegionGrid},
                               NonhydrostaticModel{<:Any, <:Any, <:AbstractArchitecture, <:MultiRegionGrid}}

# Utility to generate the inputs to complex `getregion`s
function getregionalproperties(T, inner=true) 
    type = eval(T)
    names = fieldnames(type)
    args  = Vector(undef, length(names))
    for (n, name) in enumerate(names)
        args[n] = inner ? :(_getregion(t.$name, r)) : :(getregion(t.$name, r))
    end
    return args
end

Types = (:HydrostaticFreeSurfaceModel,
         :NonhydrostaticModel,
         :ImplicitFreeSurface,
         :ExplicitFreeSurface,
         :QuasiAdamsBashforth2TimeStepper,
         :RungeKutta3TimeStepper,
         :PrescribedVelocityFields)

for T in Types
    @eval begin
        # This assumes a constructor of the form T(arg1, arg2, ...) exists,
        # # which is not the case for all types.
        @inline  getregion(t::$T, r) = $T($(getregionalproperties(T, true)...))
        @inline _getregion(t::$T, r) = $T($(getregionalproperties(T, false)...))
    end
end

@inline isregional(pv::PrescribedVelocityFields) = isregional(pv.u) | isregional(pv.v) | isregional(pv.w)
@inline devices(pv::PrescribedVelocityFields)    = devices(pv[findfirst(isregional, (pv.u, pv.v, pv.w))])

validate_tracer_advection(tracer_advection::MultiRegionObject, grid::MultiRegionGrid) = tracer_advection, NamedTuple()

@inline isregional(mrm::MultiRegionModel)        = true
@inline devices(mrm::MultiRegionModel)           = devices(mrm.grid)
@inline getdevice(mrm::MultiRegionModel, d)      = getdevice(mrm.grid, d)
@inline switch_region!(mrm::MultiRegionModel, d) = switch_region!(mrm.grid, d)

implicit_diffusion_solver(time_discretization::VerticallyImplicitTimeDiscretization, mrg::MultiRegionGrid) =
      construct_regionally(implicit_diffusion_solver, time_discretization, mrg)

WENO(mrg::MultiRegionGrid, args...; kwargs...) = construct_regionally(WENO, mrg, args...; kwargs...)

function accurate_cell_advection_timescale(grid::MultiRegionGrid, velocities)
    Δt = construct_regionally(accurate_cell_advection_timescale, grid, velocities)
    return minimum(Δt.regions)
end

function new_time_step(old_Δt, wizard, model::MultiRegionModel)
    Δt = construct_regionally(new_time_step, old_Δt, wizard, model)
    return minimum(Δt.regions)
end

ab2_step!(model::NonhydrostaticModel{<:Any, <:Any, <:Any, <:MultiRegionGrid}, Δt, χ) = @apply_regionally ab2_step!(model, Δt, χ)