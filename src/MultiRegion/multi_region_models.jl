using Oceananigans.Advection: Advection, WENO, VectorInvariant, adapt_advection_order, cell_advection_timescale, materialize_advection
using Oceananigans.BuoyancyFormulations: BuoyancyFormulations, BuoyancyForce, NegativeZDirection, AbstractBuoyancyFormulation, validate_unit_vector
using Oceananigans.TimeSteppers: TimeSteppers, QuasiAdamsBashforth2TimeStepper
using Oceananigans.Models: Models, ExplicitFreeSurface, HydrostaticFreeSurfaceModel, ImplicitFreeSurface, PrescribedVelocityFields
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModels
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: SplitExplicitFreeSurfaces,
                                                                                  FillHaloSplitExplicit,
                                                                                  apply_barotropic_kernel!,
                                                                                  _split_explicit_barotropic_velocity!,
                                                                                  _split_explicit_free_surface!

using Oceananigans.TurbulenceClosures: TurbulenceClosures, VerticallyImplicitTimeDiscretization, implicit_diffusion_solver
using Oceananigans.Advection: OnlySelfUpwinding, CrossAndSelfUpwinding
using Oceananigans.ImmersedBoundaries: GridFittedBottom, PartialCellBottom, GridFittedBoundary
using Oceananigans.Solvers: ConjugateGradientSolver
using Oceananigans.Utils: configure_kernel
using KernelAbstractions.Extras.LoopInfo: @unroll

const MultiRegionModel = HydrostaticFreeSurfaceModel{<:Any, <:Any, <:AbstractArchitecture, <:Any, <:MultiRegionGrids}
const CubedSphereModel = HydrostaticFreeSurfaceModel{<:Any, <:Any, <:AbstractArchitecture, <:Any, <:ConformalCubedSphereGridOfSomeKind}

function Advection.adapt_advection_order(advection::MultiRegionObject, grid::MultiRegionGrids)
    @apply_regionally new_advection = adapt_advection_order(advection, grid)
    return new_advection
end

function Advection.materialize_advection(advection::MultiRegionObject, grid::MultiRegionGrids)
    @apply_regionally new_advection = materialize_advection(advection, grid)
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
        @inline  Utils.getregion(t::$T, r) = $T($(getregionalproperties(T, true)...))
        @inline Utils._getregion(t::$T, r) = $T($(getregionalproperties(T, false)...))
    end
end

@inline Utils._getregion(fs::SplitExplicitFreeSurface{E}, r) where {E} =
    SplitExplicitFreeSurface{E}(getregion(fs.displacement, r),
                                getregion(fs.barotropic_velocities, r),
                                getregion(fs.filtered_state, r),
                                getregion(fs.gravitational_acceleration, r),
                                getregion(fs.kernel_parameters, r),
                                getregion(fs.substepping, r),
                                getregion(fs.timestepper, r))

@inline Utils.getregion(fs::SplitExplicitFreeSurface{E}, r) where {E} =
    SplitExplicitFreeSurface{E}(_getregion(fs.displacement, r),
                                _getregion(fs.barotropic_velocities, r),
                                _getregion(fs.filtered_state, r),
                                _getregion(fs.gravitational_acceleration, r),
                                _getregion(fs.kernel_parameters, r),
                                _getregion(fs.substepping, r),
                                _getregion(fs.timestepper, r))

# TODO: For the moment, buoyancy gradients cannot be precomputed in MultiRegionModels
function BuoyancyFormulations.BuoyancyForce(grid::MultiRegionGrids, formulation::AbstractBuoyancyFormulation;
                                            gravity_unit_vector=NegativeZDirection(),
                                            materialize_gradients=false)

    gravity_unit_vector = validate_unit_vector(gravity_unit_vector)
    return BuoyancyForce(formulation, gravity_unit_vector, nothing)
end

@inline Utils.isregional(pv::PrescribedVelocityFields) = isregional(pv.u) | isregional(pv.v) | isregional(pv.w)
@inline Utils.regions(pv::PrescribedVelocityFields)    = regions(pv[findfirst(isregional, (pv.u, pv.v, pv.w))])

HydrostaticFreeSurfaceModels.validate_tracer_advection(tracer_advection::MultiRegionObject, grid::MultiRegionGrids) = tracer_advection, NamedTuple()

# reconcile_state! for a multi-region model.
# A cubed-sphere grid needs to fill u and v velocity halos together.
function TimeSteppers.reconcile_state!(model::MultiRegionModel)

    u = model.velocities.u
    v = model.velocities.v

    fill_halo_regions!((u, v), model.clock, Oceananigans.fields(model))
    fields = Oceananigans.prognostic_fields(model)

    for key in keys(fields)
        !(key ∈ (:u, :v, :U, :V)) && fill_halo_regions!(fields[key], model.clock, Oceananigans.fields(model))
    end

    Models.HydrostaticFreeSurfaceModels.reconcile_free_surface!(model.free_surface, model.grid, model.clock, model.velocities)
    Models.HydrostaticFreeSurfaceModels.reconcile_vertical_coordinate!(model.vertical_coordinate, model, model.grid)
    return nothing
end

@inline Utils.isregional(mrm::MultiRegionModel) = true
@inline Utils.regions(mrm::MultiRegionModel) = regions(mrm.grid)

Oceananigans.TimeSteppers.cache_previous_tendencies!(model::MultiRegionModel) =
    @apply_regionally Oceananigans.TimeSteppers.cache_previous_tendencies!(model)

Oceananigans.TimeSteppers.cache_current_fields!(model::MultiRegionModel) =
    @apply_regionally Oceananigans.TimeSteppers.cache_current_fields!(model)

TurbulenceClosures.implicit_diffusion_solver(time_discretization::VerticallyImplicitTimeDiscretization, mrg::MultiRegionGrid) =
    construct_regionally(implicit_diffusion_solver, time_discretization, mrg)

Advection.WENO(mrg::MultiRegionGrid, args...; kwargs...) = construct_regionally(WENO, mrg, args...; kwargs...)

@inline Utils.getregion(t::VectorInvariant{N, FT, TD, Z, ZS, V, K, D, U, M}, r) where {N, FT, TD, Z, ZS, V, K, D, U, M} =
    VectorInvariant{N, FT, TD, M}(_getregion(t.vorticity_scheme, r),
                                  _getregion(t.vorticity_stencil, r),
                                  _getregion(t.vertical_advection_scheme, r),
                                  _getregion(t.kinetic_energy_gradient_scheme, r),
                                  _getregion(t.divergence_scheme, r),
                                  _getregion(t.upwinding, r))

@inline Utils._getregion(t::VectorInvariant{N, FT, TD, Z, ZS, V, K, D, U, M}, r) where {N, FT, TD, Z, ZS, V, K, D, U, M} =
    VectorInvariant{N, FT, TD, M}(getregion(t.vorticity_scheme, r),
                                  getregion(t.vorticity_stencil, r),
                                  getregion(t.vertical_advection_scheme, r),
                                  getregion(t.kinetic_energy_gradient_scheme, r),
                                  getregion(t.divergence_scheme, r),
                                  getregion(t.upwinding, r))

function Advection.cell_advection_timescale(grid::MultiRegionGrids, velocities)
    Δt = construct_regionally(cell_advection_timescale, grid, velocities)
    return minimum(Δt.regional_objects)
end

## Split explicit extension for complete halo filling
function SplitExplicitFreeSurfaces.iterate_split_explicit!(free_surface::FillHaloSplitExplicit, grid::ConformalCubedSphereGridOfSomeKind, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, ::Val{Nsubsteps}) where Nsubsteps
    arch = architecture(grid)

    η           = free_surface.displacement
    grid        = free_surface.displacement.grid
    state       = free_surface.filtered_state
    timestepper = free_surface.timestepper
    g           = free_surface.gravitational_acceleration
    parameters  = free_surface.kernel_parameters

    # Unpack state quantities, parameters and forcing terms.
    U, V    = free_surface.barotropic_velocities
    η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅
    Ũ, Ṽ    = state.Ũ, state.Ṽ

    @apply_regionally velocity_kernel!, _     = configure_kernel(arch, grid, parameters, _split_explicit_barotropic_velocity!)
    @apply_regionally free_surface_kernel!, _ = configure_kernel(arch, grid, parameters, _split_explicit_free_surface!)

    U_args = (grid, Val(true), Δτᴮ, η, U, V, GUⁿ, GVⁿ, g, Ũ, Ṽ, timestepper)
    η_args = (grid, Val(true), Δτᴮ, η, U, V, F, clock, η̅, U̅, V̅, timestepper)

    @unroll for substep in 1:Nsubsteps
        @inbounds averaging_weight = weights[substep]
        @inbounds transport_weight = transport_weights[substep]

        fill_halo_regions!(η)
        @apply_regionally apply_barotropic_kernel!(velocity_kernel!, transport_weight, U_args)

        fill_halo_regions!((U, V))
        @apply_regionally apply_barotropic_kernel!(free_surface_kernel!, averaging_weight, η_args)
    end

    return nothing
end
