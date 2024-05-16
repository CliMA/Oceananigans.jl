module CATKEVerticalDiffusivities

using Adapt
using KernelAbstractions: @kernel, @index

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Utils
using Oceananigans.Units
using Oceananigans.Fields
using Oceananigans.Operators

using Oceananigans.Utils: prettysummary
using Oceananigans.Grids: peripheral_node, inactive_node, inactive_cell
using Oceananigans.Fields: ZeroField
using Oceananigans.BoundaryConditions: default_prognostic_bc, DefaultBoundaryCondition
using Oceananigans.BoundaryConditions: BoundaryCondition, FieldBoundaryConditions
using Oceananigans.BoundaryConditions: DiscreteBoundaryFunction, FluxBoundaryCondition
using Oceananigans.BuoyancyModels: ∂z_b, top_buoyancy_flux
using Oceananigans.Grids: inactive_cell

using Oceananigans.TurbulenceClosures:
    getclosure,
    time_discretization,
    AbstractScalarDiffusivity,
    VerticallyImplicitTimeDiscretization,
    VerticalFormulation
    
import Oceananigans.BoundaryConditions: getbc
import Oceananigans.Utils: with_tracers
import Oceananigans.TurbulenceClosures:
    validate_closure,
    shear_production,
    buoyancy_flux,
    dissipation,
    add_closure_specific_boundary_conditions,
    compute_diffusivities!,
    DiffusivityFields,
    implicit_linear_coefficient,
    viscosity,
    diffusivity,
    viscosity_location,
    diffusivity_location,
    diffusive_flux_x,
    diffusive_flux_y,
    diffusive_flux_z

const c = Center()
const f = Face()

# @inline ℑbzᵃᵃᶜ(i, j, k, grid, fᵃᵃᶠ, args...) = ℑzᵃᵃᶜ(i, j, k, grid, fᵃᵃᶠ, args...)

# A particular kind of reconstruction that ignores peripheral nodes
@inline function ℑbzᵃᵃᶜ(i, j, k, grid, fᵃᵃᶠ, args...)
    k⁺ = k + 1
    k⁻ = k

    f⁺ = fᵃᵃᶠ(i, j, k⁺, grid, args...)
    f⁻ = fᵃᵃᶠ(i, j, k⁻, grid, args...)

    p⁺ = peripheral_node(i, j, k⁺, grid, c, c, f)
    p⁻ = peripheral_node(i, j, k⁻, grid, c, c, f)

    f⁺ = ifelse(p⁺, f⁻, f⁺)
    f⁻ = ifelse(p⁻, f⁺, f⁻)

    return (f⁺ + f⁻) / 2
end

struct CATKEVerticalDiffusivity{TD, CL, FT, DT, TKE} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 2}
    mixing_length :: CL
    turbulent_kinetic_energy_equation :: TKE
    maximum_tracer_diffusivity :: FT
    maximum_tke_diffusivity :: FT
    maximum_viscosity :: FT
    minimum_turbulent_kinetic_energy :: FT
    minimum_convective_buoyancy_flux :: FT
    negative_turbulent_kinetic_energy_damping_time_scale :: FT
    turbulent_kinetic_energy_time_step :: DT
end

function CATKEVerticalDiffusivity{TD}(mixing_length::CL,
                                      turbulent_kinetic_energy_equation::TKE,
                                      maximum_tracer_diffusivity::FT,
                                      maximum_tke_diffusivity::FT,
                                      maximum_viscosity::FT,
                                      minimum_turbulent_kinetic_energy::FT,
                                      minimum_convective_buoyancy_flux::FT,
                                      negative_turbulent_kinetic_energy_damping_time_scale::FT, 
                                      turbulent_kinetic_energy_time_step::DT) where {TD, CL, FT, DT, TKE}

    return CATKEVerticalDiffusivity{TD, CL, FT, DT, TKE}(mixing_length,
                                                         turbulent_kinetic_energy_equation,
                                                         maximum_tracer_diffusivity,
                                                         maximum_tke_diffusivity,
                                                         maximum_viscosity,
                                                         minimum_turbulent_kinetic_energy,
                                                         minimum_convective_buoyancy_flux,
                                                         negative_turbulent_kinetic_energy_damping_time_scale,
                                                         turbulent_kinetic_energy_time_step)
end

CATKEVerticalDiffusivity(FT::DataType; kw...) =
    CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

const CATKEVD{TD} = CATKEVerticalDiffusivity{TD} where TD
const CATKEVDArray{TD} = AbstractArray{<:CATKEVD{TD}} where TD
const FlavorOfCATKE{TD} = Union{CATKEVD{TD}, CATKEVDArray{TD}} where TD

include("mixing_length.jl")
include("turbulent_kinetic_energy_equation.jl")
include("time_step_turbulent_kinetic_energy.jl")

"""
    CATKEVerticalDiffusivity([time_discretization = VerticallyImplicitTimeDiscretization(),
                             FT = Float64;]
                             mixing_length = MixingLength(),
                             turbulent_kinetic_energy_equation = TurbulentKineticEnergyEquation(),
                             maximum_tracer_diffusivity = Inf,
                             maximum_tke_diffusivity = Inf,
                             maximum_viscosity = Inf,
                             minimum_turbulent_kinetic_energy = 1e-9,
                             minimum_convective_buoyancy_flux = 1e-11,
                             negative_turbulent_kinetic_energy_damping_time_scale = 1minute,
                             turbulent_kinetic_energy_time_step = nothing)

Return the `CATKEVerticalDiffusivity` turbulence closure for vertical mixing by
small-scale ocean turbulence based on the prognostic evolution of subgrid
Turbulent Kinetic Energy (TKE).

!!! note "CATKE vertical diffusivity"
    `CATKEVerticalDiffusivity` is new turbulence closure diffusivity. The default
    values for its free parameters are obtained from calibration against large eddy
    simulations. For more details please refer to [Wagner23catke](@cite).

    Use with caution and report any issues with the physics at https://github.com/CliMA/Oceananigans.jl/issues.

Arguments
=========

- `time_discretization`: Either `ExplicitTimeDiscretization()` or `VerticallyImplicitTimeDiscretization()`;
                         default `VerticallyImplicitTimeDiscretization()`.

- `FT`: Float type; default `Float64`.


Keyword arguments
=================

- `maximum_diffusivity`: Maximum value for tracer, momentum, and TKE diffusivities.
                        Used to clip the diffusivity when/if CATKE predicts
                        diffusivities that are too large.
                        Default: `Inf`.

- `minimum_turbulent_kinetic_energy`: Minimum value for the turbulent kinetic energy.
                                    Can be used to model the presence "background" TKE
                                    levels due to, for example, mixing by breaking internal waves.
                                    Default: 0.

- `negative_turbulent_kinetic_energy_damping_time_scale`: Damping time-scale for spurious negative values of TKE,
                                                        typically generated by oscillatory errors associated
                                                        with TKE advection.
                                                        Default: 1 minute.

Note that for numerical stability, it is recommended to either have a relative short
`negative_turbulent_kinetic_energy_damping_time_scale` or a reasonable
`minimum_turbulent_kinetic_energy`, or both.
"""
function CATKEVerticalDiffusivity(time_discretization::TD = VerticallyImplicitTimeDiscretization(),
                                  FT = Float64;
                                  mixing_length = MixingLength(),
                                  turbulent_kinetic_energy_equation = TurbulentKineticEnergyEquation(),
                                  maximum_tracer_diffusivity = Inf,
                                  maximum_tke_diffusivity = Inf,
                                  maximum_viscosity = Inf,
                                  minimum_turbulent_kinetic_energy = 1e-9,
                                  minimum_convective_buoyancy_flux = 1e-11,
                                  negative_turbulent_kinetic_energy_damping_time_scale = 1minute,
                                  turbulent_kinetic_energy_time_step = nothing) where TD

    mixing_length = convert_eltype(FT, mixing_length)
    turbulent_kinetic_energy_equation = convert_eltype(FT, turbulent_kinetic_energy_equation)

    return CATKEVerticalDiffusivity{TD}(mixing_length,
                                        turbulent_kinetic_energy_equation,
                                        convert(FT, maximum_tracer_diffusivity),
                                        convert(FT, maximum_tke_diffusivity),
                                        convert(FT, maximum_viscosity),
                                        convert(FT, minimum_turbulent_kinetic_energy),
                                        convert(FT, minimum_convective_buoyancy_flux),
                                        convert(FT, negative_turbulent_kinetic_energy_damping_time_scale),
                                        turbulent_kinetic_energy_time_step)
end

function with_tracers(tracer_names, closure::FlavorOfCATKE)
    :e ∈ tracer_names ||
        throw(ArgumentError("Tracers must contain :e to represent turbulent kinetic energy " *
                            "for `CATKEVerticalDiffusivity`."))

    return closure
end

# For tuples of closures, we need to know _which_ closure is CATKE.
# Here we take a "simple" approach that sorts the tuple so CATKE is first.
# This is not sustainable though if multiple closures require this.
# The two other possibilities are:
# 1. Recursion to find which closure is CATKE in a compiler-inferrable way
# 2. Store the "CATKE index" inside CATKE via validate_closure.
validate_closure(closure_tuple::Tuple) = Tuple(sort(collect(closure_tuple), lt=catke_first))

catke_first(closure1, catke::FlavorOfCATKE) = false
catke_first(catke::FlavorOfCATKE, closure2) = true
catke_first(closure1, closure2) = false
catke_first(catke1::FlavorOfCATKE, catke2::FlavorOfCATKE) = error("Can't have two CATKEs in one closure tuple.")

#####
##### Mixing length and TKE equation
#####

@inline Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy) =
    ℑbzᵃᵃᶜ(i, j, k, grid, Riᶜᶜᶠ, velocities, tracers, buoyancy)

@inline function Riᶜᶜᶠ(i, j, k, grid, velocities, tracers, buoyancy)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    S² = ∂z_u² + ∂z_v²
    Ri = N² / S²
    #return ifelse(N² ≤ 0, zero(grid), Ri)
    return ifelse(N² == 0, zero(grid), Ri)
end

for S in (:MixingLength, :TurbulentKineticEnergyEquation)
    @eval @inline convert_eltype(::Type{FT}, s::$S) where FT =
        $S{FT}(; Dict(p => getproperty(s, p) for p in propertynames(s))...)
    @eval @inline convert_eltype(::Type{FT}, s::$S{FT}) where FT = s
end

#####
##### Diffusivities and diffusivity fields utilities
#####

function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfCATKE)

    default_diffusivity_bcs = (κu = FieldBoundaryConditions(grid, (Center, Center, Face)),
                               κc = FieldBoundaryConditions(grid, (Center, Center, Face)),
                               κe = FieldBoundaryConditions(grid, (Center, Center, Face)))

    bcs = merge(default_diffusivity_bcs, bcs)

    κu = ZFaceField(grid, boundary_conditions=bcs.κu)
    κc = ZFaceField(grid, boundary_conditions=bcs.κc)
    κe = ZFaceField(grid, boundary_conditions=bcs.κe)
    Le = CenterField(grid)
    Jᵇ = Field{Center, Center, Nothing}(grid)
    previous_compute_time = Ref(zero(grid))

    # Note: we may be able to avoid using the "previous velocities" in favor of a "fully implicit"
    # discretization of shear production
    u⁻ = XFaceField(grid)
    v⁻ = YFaceField(grid)
    previous_velocities = (; u=u⁻, v=v⁻)

    # Secret tuple for getting tracer diffusivities with tuple[tracer_index]
    _tupled_tracer_diffusivities         = NamedTuple(name => name === :e ? κe : κc          for name in tracer_names)
    _tupled_implicit_linear_coefficients = NamedTuple(name => name === :e ? Le : ZeroField() for name in tracer_names)

    return (; κu, κc, κe, Le, Jᵇ,
            previous_compute_time, previous_velocities,
            _tupled_tracer_diffusivities, _tupled_implicit_linear_coefficients)
end        

const c = Center()
const f = Face()

@inline viscosity_location(::FlavorOfCATKE) = (c, c, f)
@inline diffusivity_location(::FlavorOfCATKE) = (c, c, f)
@inline clip(x) = max(zero(x), x)

function compute_diffusivities!(diffusivities, closure::FlavorOfCATKE, model; parameters = :xyz)

    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = model.tracers
    buoyancy = model.buoyancy
    clock = model.clock
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))
    Δt = model.clock.time - diffusivities.previous_compute_time[]
    diffusivities.previous_compute_time[] = model.clock.time

    if isfinite(model.clock.last_Δt) # Check that we have taken a valid time-step first.
        # Compute e at the current time:
        #   * update tendency Gⁿ using current and previous velocity field
        #   * use tridiagonal solve to take an implicit step
        time_step_turbulent_kinetic_energy!(model)
    end

    # Update "previous velocities"
    u, v, w = model.velocities
    u⁻, v⁻ = diffusivities.previous_velocities
    parent(u⁻) .= parent(u)
    parent(v⁻) .= parent(v)

    launch!(arch, grid, :xy,
            compute_average_surface_buoyancy_flux!,
            diffusivities.Jᵇ, grid, closure, velocities, tracers, buoyancy, top_tracer_bcs, clock, Δt)

    launch!(arch, grid, parameters,
            compute_CATKE_diffusivities!,
            diffusivities, grid, closure, velocities, tracers, buoyancy)

    return nothing
end

@kernel function compute_average_surface_buoyancy_flux!(Jᵇ, grid, closure, velocities, tracers,
                                                        buoyancy, top_tracer_bcs, clock, Δt)
    i, j = @index(Global, NTuple)
    k = grid.Nz

    closure = getclosure(i, j, closure)

    model_fields = merge(velocities, tracers)
    Jᵇ★ = top_buoyancy_flux(i, j, grid, buoyancy, top_tracer_bcs, clock, model_fields)
    ℓᴰ = dissipation_length_scaleᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, Jᵇ)

    Jᵇᵋ = closure.minimum_convective_buoyancy_flux
    Jᵇᵢⱼ = @inbounds Jᵇ[i, j, 1]
    Jᵇ⁺ = max(Jᵇᵋ, Jᵇᵢⱼ, Jᵇ★) # selects fastest (dominant) time-scale
    t★ = (ℓᴰ^2 / Jᵇ⁺)^(1/3)
    ϵ = Δt / t★

    @inbounds Jᵇ[i, j, 1] = (Jᵇᵢⱼ + ϵ * Jᵇ★) / (1 + ϵ)
end

@inline function mask_diffusivity(i, j, k, grid, κ★)
    on_periphery = peripheral_node(i, j, k, grid, c, c, f)
    within_inactive = inactive_node(i, j, k, grid, c, c, f)
    nan = convert(eltype(grid), NaN)
    return ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κ★))
end

@kernel function compute_CATKE_diffusivities!(diffusivities, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)
    Jᵇ = diffusivities.Jᵇ

    # Note: we also compute the TKE diffusivity here for diagnostic purposes, even though it
    # is recomputed in time_step_turbulent_kinetic_energy.
    κu★ = κuᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, Jᵇ)
    κc★ = κcᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, Jᵇ)
    κe★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, Jᵇ)

    κu★ = mask_diffusivity(i, j, k, grid, κu★)
    κc★ = mask_diffusivity(i, j, k, grid, κc★)
    κe★ = mask_diffusivity(i, j, k, grid, κe★)

    @inbounds begin
        diffusivities.κu[i, j, k] = κu★
        diffusivities.κc[i, j, k] = κc★
        diffusivities.κe[i, j, k] = κe★
    end
end

@inline function turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)
    eᵢ = @inbounds e[i, j, k]
    eᵐⁱⁿ = closure.minimum_turbulent_kinetic_energy
    return sqrt(max(eᵐⁱⁿ, eᵢ))
end

@inline function κuᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, surface_buoyancy_flux)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓu = momentum_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, surface_buoyancy_flux)
    κu = ℓu * w★
    κu_max = closure.maximum_viscosity
    return min(κu, κu_max)
end

@inline function κcᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, surface_buoyancy_flux)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓc = tracer_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, surface_buoyancy_flux)
    κc = ℓc * w★
    κc_max = closure.maximum_tracer_diffusivity
    return min(κc, κc_max)
end

@inline function κeᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, surface_buoyancy_flux)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓe = TKE_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, surface_buoyancy_flux)
    κe = ℓe * w★
    κe_max = closure.maximum_tke_diffusivity
    return min(κe, κe_max)
end

@inline viscosity(::FlavorOfCATKE, diffusivities) = diffusivities.κu
@inline diffusivity(::FlavorOfCATKE, diffusivities, ::Val{id}) where id = diffusivities._tupled_tracer_diffusivities[id]
    
#####
##### Show
#####

function Base.summary(closure::CATKEVD)
    TD = nameof(typeof(time_discretization(closure)))
    return string("CATKEVerticalDiffusivity{$TD}")
end

function Base.show(io::IO, clo::FlavorOfCATKE)
    print(io, summary(clo))
    print(io, '\n')
    print(io, "├── maximum_tracer_diffusivity: ", prettysummary(clo.maximum_tracer_diffusivity), '\n',
              "├── maximum_tke_diffusivity: ", prettysummary(clo.maximum_tke_diffusivity), '\n',
              "├── maximum_viscosity: ", prettysummary(clo.maximum_viscosity), '\n',
              "├── minimum_turbulent_kinetic_energy: ", prettysummary(clo.minimum_turbulent_kinetic_energy), '\n',
              "├── negative_turbulent_kinetic_energy_damping_time_scale: ", prettysummary(clo.negative_turbulent_kinetic_energy_damping_time_scale), '\n',
              "├── minimum_convective_buoyancy_flux: ", prettysummary(clo.minimum_convective_buoyancy_flux), '\n',
              "├── turbulent_kinetic_energy_time_step: ", prettysummary(clo.turbulent_kinetic_energy_time_step), '\n',
              "├── mixing_length: ", prettysummary(clo.mixing_length), '\n',
              "│   ├── Cˢ:   ", prettysummary(clo.mixing_length.Cˢ), '\n',
              "│   ├── Cᵇ:   ", prettysummary(clo.mixing_length.Cᵇ), '\n',
              "│   ├── Cʰⁱu: ", prettysummary(clo.mixing_length.Cʰⁱu), '\n',
              "│   ├── Cʰⁱc: ", prettysummary(clo.mixing_length.Cʰⁱc), '\n',
              "│   ├── Cʰⁱe: ", prettysummary(clo.mixing_length.Cʰⁱe), '\n',
              "│   ├── Cˡᵒu: ", prettysummary(clo.mixing_length.Cˡᵒu), '\n',
              "│   ├── Cˡᵒc: ", prettysummary(clo.mixing_length.Cˡᵒc), '\n',
              "│   ├── Cˡᵒe: ", prettysummary(clo.mixing_length.Cˡᵒe), '\n',
              "│   ├── Cᵘⁿu: ", prettysummary(clo.mixing_length.Cᵘⁿu), '\n',
              "│   ├── Cᵘⁿc: ", prettysummary(clo.mixing_length.Cᵘⁿc), '\n',
              "│   ├── Cᵘⁿe: ", prettysummary(clo.mixing_length.Cᵘⁿe), '\n',
              "│   ├── Cᶜu:  ", prettysummary(clo.mixing_length.Cᶜu), '\n',
              "│   ├── Cᶜc:  ", prettysummary(clo.mixing_length.Cᶜc), '\n',
              "│   ├── Cᶜe:  ", prettysummary(clo.mixing_length.Cᶜe), '\n',
              "│   ├── Cᵉc:  ", prettysummary(clo.mixing_length.Cᵉc), '\n',
              "│   ├── Cᵉe:  ", prettysummary(clo.mixing_length.Cᵉe), '\n',
              "│   ├── Cˢᵖ:  ", prettysummary(clo.mixing_length.Cˢᵖ), '\n',
              "│   ├── CRiᵟ: ", prettysummary(clo.mixing_length.CRiᵟ), '\n',
              "│   └── CRi⁰: ", prettysummary(clo.mixing_length.CRi⁰), '\n',
              "└── turbulent_kinetic_energy_equation: ", prettysummary(clo.turbulent_kinetic_energy_equation), '\n',
              "    ├── CʰⁱD: ", prettysummary(clo.turbulent_kinetic_energy_equation.CʰⁱD),  '\n',
              "    ├── CˡᵒD: ", prettysummary(clo.turbulent_kinetic_energy_equation.CˡᵒD),  '\n',
              "    ├── CᵘⁿD: ", prettysummary(clo.turbulent_kinetic_energy_equation.CᵘⁿD),  '\n',
              "    ├── CᶜD:  ", prettysummary(clo.turbulent_kinetic_energy_equation.CᶜD),  '\n',
              "    ├── CᵉD:  ", prettysummary(clo.turbulent_kinetic_energy_equation.CᵉD),  '\n',
              "    ├── Cᵂu★: ", prettysummary(clo.turbulent_kinetic_energy_equation.Cᵂu★), '\n',
              "    ├── CᵂwΔ: ", prettysummary(clo.turbulent_kinetic_energy_equation.CᵂwΔ), '\n',
              "    └── Cᵂϵ:  ", prettysummary(clo.turbulent_kinetic_energy_equation.Cᵂϵ))
end

end # module

