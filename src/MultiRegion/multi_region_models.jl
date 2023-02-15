using Oceananigans.Models: AbstractModel
using Oceananigans.Advection: WENO, VectorInvariant
using Oceananigans.Models.HydrostaticFreeSurfaceModels: AbstractFreeSurface
using Oceananigans.TimeSteppers: AbstractTimeStepper, QuasiAdamsBashforth2TimeStepper
using Oceananigans.Models: PrescribedVelocityFields
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.Advection: AbstractAdvectionScheme

import Oceananigans.Simulations: new_time_step
import Oceananigans.Diagnostics: accurate_cell_advection_timescale
import Oceananigans.Advection: WENO
import Oceananigans.Models.HydrostaticFreeSurfaceModels: build_implicit_step_solver, validate_tracer_advection
import Oceananigans.TurbulenceClosures: implicit_diffusion_solver

const MultiRegionModel = HydrostaticFreeSurfaceModel{<:Any, <:Any, <:AbstractArchitecture, <:Any, <:MultiRegionGrid}

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
         :ImplicitFreeSurface,
         :ExplicitFreeSurface,
         :QuasiAdamsBashforth2TimeStepper,
         :SplitExplicitAuxiliary,
         :SplitExplicitState,
         :SplitExplicitFreeSurface,
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

implicit_diffusion_solver(time_discretization::VerticallyImplicitTimeDiscretization, mrg::MultiRegionGrid) =
      construct_regionally(implicit_diffusion_solver, time_discretization, mrg)

WENO(mrg::MultiRegionGrid, args...; kwargs...) = construct_regionally(WENO, mrg, args...; kwargs...)

@inline  getregion(t::VectorInvariant{N, FT}, r) where {N, FT} = 
                VectorInvariant{N, FT}(_getregion(t.vorticity_scheme, r), 
                                       _getregion(t.divergence_scheme, r),
                                       t.vorticity_stencil, 
                                       t.divergence_stencil, 
                                       _getregion(t.vertical_scheme, r))

@inline _getregion(t::VectorInvariant{N, FT}, r) where {N, FT} = 
                VectorInvariant{N, FT}(getregion(t.vorticity_scheme, r), 
                                       getregion(t.divergence_scheme, r), 
                                       t.vorticity_stencil, 
                                       t.divergence_stencil, 
                                       getregion(t.vertical_scheme, r))

function accurate_cell_advection_timescale(grid::MultiRegionGrid, velocities)
    Δt = construct_regionally(accurate_cell_advection_timescale, grid, velocities)
    return minimum(Δt.regional_objects)
end

function new_time_step(old_Δt, wizard, model::MultiRegionModel)
    Δt = construct_regionally(new_time_step, old_Δt, wizard, model)
    return minimum(Δt.regional_objects)
end
