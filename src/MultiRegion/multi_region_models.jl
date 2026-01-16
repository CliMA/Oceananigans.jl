using Oceananigans.Advection: WENO, VectorInvariant
using Oceananigans.BuoyancyFormulations: NegativeZDirection, AbstractBuoyancyFormulation, validate_unit_vector
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, update_state!
using Oceananigans.Models: PrescribedVelocityFields
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.Advection: OnlySelfUpwinding, CrossAndSelfUpwinding
using Oceananigans.ImmersedBoundaries: GridFittedBottom, PartialCellBottom, GridFittedBoundary
using Oceananigans.Solvers: ConjugateGradientSolver

import Oceananigans.BuoyancyFormulations: BuoyancyForce
import Oceananigans.Advection: WENO, cell_advection_timescale, adapt_advection_order
import Oceananigans.BuoyancyFormulations: BuoyancyForce
import Oceananigans.Models: initialization_update_state!
import Oceananigans.Models.HydrostaticFreeSurfaceModels: validate_tracer_advection
import Oceananigans.TurbulenceClosures: implicit_diffusion_solver

const MultiRegionModel = HydrostaticFreeSurfaceModel{<:Any, <:Any, <:AbstractArchitecture, <:Any, <:MultiRegionGrids}
const CubedSphereModel = HydrostaticFreeSurfaceModel{<:Any, <:Any, <:AbstractArchitecture, <:Any, <:ConformalCubedSphereGridOfSomeKind}

function adapt_advection_order(advection::MultiRegionObject, grid::MultiRegionGrids)
    @apply_regionally new_advection = adapt_advection_order(advection, grid)
    return new_advection
end

# Utility to generate the inputs to complex `getregion`s
function getregionalproperties(type, inner=true)
    names = fieldnames(type)
    args  = Vector(undef, length(names))
    for (n, name) in enumerate(names)
        args[n] = inner ? :(_getregion(t.$name, r)) : :(getregion(t.$name, r))
    end
    return args
end

Types = (HydrostaticFreeSurfaceModel,
         ImplicitFreeSurface,
         ExplicitFreeSurface,
         QuasiAdamsBashforth2TimeStepper,
         SplitExplicitFreeSurface,
         PrescribedVelocityFields,
         ConjugateGradientSolver,
         CrossAndSelfUpwinding,
         OnlySelfUpwinding,
         GridFittedBoundary,
         GridFittedBottom,
         PartialCellBottom)

for T in Types
    @eval begin
        # This assumes a constructor of the form T(arg1, arg2, ...) exists,
        # which is not the case for all types.
        @inline  getregion(t::$T, r) = $T($(getregionalproperties(T, true)...))
        @inline _getregion(t::$T, r) = $T($(getregionalproperties(T, false)...))
    end
end

# TODO: For the moment, buoyancy gradients cannot be precomputed in MultiRegionModels
function BuoyancyForce(grid::MultiRegionGrids, formulation::AbstractBuoyancyFormulation;
                       gravity_unit_vector=NegativeZDirection(),
                       materialize_gradients=false)

    gravity_unit_vector = validate_unit_vector(gravity_unit_vector)
    return BuoyancyForce(formulation, gravity_unit_vector, nothing)
end

@inline isregional(pv::PrescribedVelocityFields) = isregional(pv.u) | isregional(pv.v) | isregional(pv.w)
@inline regions(pv::PrescribedVelocityFields)    = regions(pv[findfirst(isregional, (pv.u, pv.v, pv.w))])

validate_tracer_advection(tracer_advection::MultiRegionObject, grid::MultiRegionGrids) = tracer_advection, NamedTuple()

# A cubed sphere needs to fill u and v separately
# U and V (in case of a `SplitExplicitFreeSurface`) are filled in `initialize!`
function initialization_update_state!(model::CubedSphereModel)

    # Update the state of the model
    update_state!(model)

    u = model.velocities.u
    v = model.velocities.v

    fill_halo_regions!((u, v), model.clock, Oceananigans.fields(model))
    fields = Oceananigans.prognostic_fields(model)

    for key in keys(fields)
        !(key ∈ (:u, :v, :U, :V)) && fill_halo_regions!(fields[key], model.clock, Oceananigans.fields(model))
    end

    # Finally, initialize the model (e.g., free surface, vertical coordinate...)
    Oceananigans.initialize!(model)

    return nothing
end

@inline isregional(mrm::MultiRegionModel) = true
@inline regions(mrm::MultiRegionModel) = regions(mrm.grid)

Oceananigans.TimeSteppers.cache_previous_tendencies!(model::MultiRegionModel) = 
    @apply_regionally Oceananigans.TimeSteppers.cache_previous_tendencies!(model)

Oceananigans.TimeSteppers.cache_current_fields!(model::MultiRegionModel) = 
    @apply_regionally Oceananigans.TimeSteppers.cache_current_fields!(model)

implicit_diffusion_solver(time_discretization::VerticallyImplicitTimeDiscretization, mrg::MultiRegionGrid) =
    construct_regionally(implicit_diffusion_solver, time_discretization, mrg)

WENO(mrg::MultiRegionGrid, args...; kwargs...) = construct_regionally(WENO, mrg, args...; kwargs...)

@inline getregion(t::VectorInvariant{N, FT, Z, ZS, V, K, D, U, M}, r) where {N, FT, Z, ZS, V, K, D, U, M} =
    VectorInvariant{N, FT, M}(_getregion(t.vorticity_scheme, r),
                              _getregion(t.vorticity_stencil, r),
                              _getregion(t.vertical_advection_scheme, r),
                              _getregion(t.kinetic_energy_gradient_scheme, r),
                              _getregion(t.divergence_scheme, r),
                              _getregion(t.upwinding, r))

@inline _getregion(t::VectorInvariant{N, FT, Z, ZS, V, K, D, U, M}, r) where {N, FT, Z, ZS, V, K, D, U, M} =
    VectorInvariant{N, FT, M}(getregion(t.vorticity_scheme, r),
                              getregion(t.vorticity_stencil, r),
                              getregion(t.vertical_advection_scheme, r),
                              getregion(t.kinetic_energy_gradient_scheme, r),
                              getregion(t.divergence_scheme, r),
                              getregion(t.upwinding, r))

function cell_advection_timescale(grid::MultiRegionGrids, velocities)
    Δt = construct_regionally(cell_advection_timescale, grid, velocities)
    return minimum(Δt.regional_objects)
end
