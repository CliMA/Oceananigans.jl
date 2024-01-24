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

struct CATKEVerticalDiffusivity{TD, CL, FT, TKE} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 2}
    mixing_length :: CL
    turbulent_kinetic_energy_equation :: TKE
    maximum_tracer_diffusivity :: FT
    maximum_tke_diffusivity :: FT
    maximum_viscosity :: FT
    minimum_turbulent_kinetic_energy :: FT
    minimum_convective_buoyancy_flux :: FT
    negative_turbulent_kinetic_energy_damping_time_scale :: FT
end

CATKEVerticalDiffusivity{TD}(mixing_length::CL,
                             turbulent_kinetic_energy_equation::TKE,
                             maximum_tracer_diffusivity::FT,
                             maximum_tke_diffusivity::FT,
                             maximum_viscosity::FT,
                             minimum_turbulent_kinetic_energy::FT,
                             minimum_convective_buoyancy_flux::FT,
                             negative_turbulent_kinetic_energy_damping_time_scale::FT) where {TD, CL, TKE, FT} =
    CATKEVerticalDiffusivity{TD, CL, FT, TKE}(mixing_length,
                                              turbulent_kinetic_energy_equation,
                                              maximum_tracer_diffusivity,
                                              maximum_tke_diffusivity,
                                              maximum_viscosity,
                                              minimum_turbulent_kinetic_energy,
                                              minimum_convective_buoyancy_flux,
                                              negative_turbulent_kinetic_energy_damping_time_scale)

CATKEVerticalDiffusivity(FT::DataType; kw...) =
    CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

const CATKEVD{TD} = CATKEVerticalDiffusivity{TD} where TD
const CATKEVDArray{TD} = AbstractArray{<:CATKEVD{TD}} where TD
const FlavorOfCATKE{TD} = Union{CATKEVD{TD}, CATKEVDArray{TD}} where TD

include("mixing_length.jl")
include("turbulent_kinetic_energy_equation.jl")

"""
    CATKEVerticalDiffusivity([time_discretization = VerticallyImplicitTimeDiscretization(),
                             FT = Float64;]
                             mixing_length = MixingLength(),
                             turbulent_kinetic_energy_equation = TurbulentKineticEnergyEquation(),
                             maximum_tracer_diffusivity = Inf,
                             maximum_tke_diffusivity = Inf,
                             maximum_viscosity = Inf,
                             minimum_turbulent_kinetic_energy = 1e-6,
                             minimum_convective_buoyancy_flux = 1e-8,
                             negative_turbulent_kinetic_energy_damping_time_scale = 1minute)

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
                                  minimum_turbulent_kinetic_energy = 1e-6,
                                  minimum_convective_buoyancy_flux = 1e-8,
                                  negative_turbulent_kinetic_energy_damping_time_scale = 1minute) where TD

    mixing_length = convert_eltype(FT, mixing_length)
    turbulent_kinetic_energy_equation = convert_eltype(FT, turbulent_kinetic_energy_equation)

    return CATKEVerticalDiffusivity{TD}(mixing_length,
                                        turbulent_kinetic_energy_equation,
                                        FT(maximum_tracer_diffusivity),
                                        FT(maximum_tke_diffusivity),
                                        FT(maximum_viscosity),
                                        FT(minimum_turbulent_kinetic_energy),
                                        FT(minimum_convective_buoyancy_flux),
                                        FT(negative_turbulent_kinetic_energy_damping_time_scale))
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
    ℑzᵃᵃᶜ(i, j, k, grid, Riᶜᶜᶠ, velocities, tracers, buoyancy)

@inline function Riᶜᶜᶠ(i, j, k, grid, velocities, tracers, buoyancy)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    S² = ∂z_u² + ∂z_v²
    Ri = N² / S²
    return ifelse(N² ≤ 0, zero(grid), Ri)
end

for S in (:MixingLength, :TurbulentKineticEnergyEquation)
    @eval @inline convert_eltype(::Type{FT}, s::$S) where FT = $S{FT}(; Dict(p => getproperty(s, p) for p in propertynames(s))...)
    @eval @inline convert_eltype(::Type{FT}, s::$S{FT}) where FT = s
end

#####
##### Diffusivities and diffusivity fields utilities
#####

function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfCATKE)

    default_diffusivity_bcs = (κᵘ = FieldBoundaryConditions(grid, (Center, Center, Face)),
                               κᶜ = FieldBoundaryConditions(grid, (Center, Center, Face)),
                               κᵉ = FieldBoundaryConditions(grid, (Center, Center, Face)))

    bcs = merge(default_diffusivity_bcs, bcs)

    κᵘ = ZFaceField(grid, boundary_conditions=bcs.κᵘ)
    κᶜ = ZFaceField(grid, boundary_conditions=bcs.κᶜ)
    κᵉ = ZFaceField(grid, boundary_conditions=bcs.κᵉ)
    Lᵉ = CenterField(grid)
    Qᵇ = Field{Center, Center, Nothing}(grid)
    previous_compute_time = Ref(zero(grid))

    # Secret tuple for getting tracer diffusivities with tuple[tracer_index]
    _tupled_tracer_diffusivities         = NamedTuple(name => name === :e ? κᵉ : κᶜ          for name in tracer_names)
    _tupled_implicit_linear_coefficients = NamedTuple(name => name === :e ? Lᵉ : ZeroField() for name in tracer_names)

    S² = CenterField(grid)
    N² = ZFaceField(grid)
    w★ = CenterField(grid)

    return (; κᵘ, κᶜ, κᵉ, Lᵉ, Qᵇ, S², N², w★, previous_compute_time, _tupled_tracer_diffusivities, _tupled_implicit_linear_coefficients)
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

    launch!(arch, grid, parameters,
            _compute_CATKE_auxiliaries!,
            diffusivities, grid, closure, velocities, tracers, buoyancy)

    launch!(arch, grid, :xy,
            _compute_average_surface_buoyancy_flux!,
            diffusivities.Qᵇ, grid, closure, diffusivities, velocities, tracers, buoyancy, top_tracer_bcs, clock, Δt)

    launch!(arch, grid, parameters,
            _compute_CATKE_diffusivities!,
            diffusivities, grid, closure, velocities, tracers, buoyancy)

    return nothing
end

@kernel function _compute_CATKE_auxiliaries!(diffusivities, grid, closure, velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)

    S² = diffusivities.S²
    N² = diffusivities.N²
    w★ = diffusivities.w★
    u, v, w = velocities

    @inbounds begin
        S²[i, j, k] = shearᶜᶜᶠ(i, j, k, grid, u, v)
        N²[i, j, k] = ∂z_b(i, j, k, grid, buoyancy, tracers)
        w★[i, j, k] = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, tracers.e)
    end
end

@kernel function _compute_average_surface_buoyancy_flux!(Qᵇ, grid, closure, diffusivities, velocities, tracers, buoyancy, top_tracer_bcs, clock, Δt)
    i, j = @index(Global, NTuple)

    S² = diffusivities.S²
    N² = diffusivities.N²
    w★ = diffusivities.w★

    closure = getclosure(i, j, closure)

    Qᵇ★ = top_buoyancy_flux(i, j, grid, buoyancy, top_tracer_bcs, clock, merge(velocities, tracers))

    k = grid.Nz
    ℓᴰ = dissipation_length_scaleᶜᶜᶜ(i, j, k, grid, closure, Qᵇ, S², N², w★)

    Qᵇᵋ = closure.minimum_convective_buoyancy_flux
    Qᵇᵢⱼ = @inbounds Qᵇ[i, j, 1]
    Qᵇ⁺ = max(Qᵇᵋ, Qᵇᵢⱼ, Qᵇ★) # selects fastest (dominant) time-scale
    t★ = (ℓᴰ^2 / Qᵇ⁺)^(1/3)
    ϵ = Δt / t★

    @inbounds Qᵇ[i, j, 1] = (Qᵇᵢⱼ + ϵ * Qᵇ★) / (1 + ϵ)
end

@kernel function _compute_CATKE_diffusivities!(diffusivities, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)
    Qᵇ = diffusivities.Qᵇ
    S² = diffusivities.S²
    N² = diffusivities.N²
    w★ = diffusivities.w★

    @inbounds begin
        κᵘ★ = κuᶜᶜᶠ(i, j, k, grid, closure_ij, Qᵇ, S², N², w★)
        κᶜ★ = κcᶜᶜᶠ(i, j, k, grid, closure_ij, Qᵇ, S², N², w★)
        κᵉ★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, Qᵇ, S², N², w★)

        on_periphery = peripheral_node(i, j, k, grid, c, c, f)
        within_inactive = inactive_node(i, j, k, grid, c, c, f)
        nan = convert(eltype(grid), NaN)
        κᵘ★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κᵘ★))
        κᶜ★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κᶜ★))
        κᵉ★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κᵉ★))

        diffusivities.κᵘ[i, j, k] = κᵘ★
        diffusivities.κᶜ[i, j, k] = κᶜ★
        diffusivities.κᵉ[i, j, k] = κᵉ★

        # "Patankar trick" for buoyancy production (cf Patankar 1980 or Burchard et al. 2003)
        # If buoyancy flux is a _sink_ of TKE, we treat it implicitly.
        wb = explicit_buoyancy_flux(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivities)
        eⁱʲᵏ = @inbounds tracers.e[i, j, k]

        # See `buoyancy_flux`
        dissipative_buoyancy_flux = sign(wb) * sign(eⁱʲᵏ) < 0
        wb_e = ifelse(dissipative_buoyancy_flux, wb / eⁱʲᵏ, zero(grid))

        # Implicit TKE flux at solid bottoms (extra damping for TKE near boundaries)
        on_bottom = !inactive_cell(i, j, k, grid) & inactive_cell(i, j, k-1, grid)
        Δz = Δzᶜᶜᶜ(i, j, k, grid)
        Cᵂϵ = closure_ij.turbulent_kinetic_energy_equation.Cᵂϵ
        Q_e = - Cᵂϵ * w★[i, j, k] / Δz * on_bottom

        # Implicit TKE dissipation
        ω_e = dissipation_rate(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, diffusivities)
        
        diffusivities.Lᵉ[i, j, k] = - wb_e - ω_e + Q_e
    end
end

@inline function implicit_linear_coefficient(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, K, ::Val{id}, args...) where id
    L = K._tupled_implicit_linear_coefficients[id]
    return @inbounds L[i, j, k]
end

@inline function turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)
    eᵢ = @inbounds e[i, j, k]
    eᵐⁱⁿ = closure.minimum_turbulent_kinetic_energy
    return sqrt(max(eᵐⁱⁿ, eᵢ))
end

@inline function κuᶜᶜᶠ(i, j, k, grid, closure, surface_buoyancy_flux, S², N², w★)
    w★ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, w★)
    ℓᵘ = momentum_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, S², N², w★)
    κᵘ = ℓᵘ * w★ᶜᶜᶠ
    κᵘ_max = closure.maximum_viscosity
    return min(κᵘ, κᵘ_max)
end

@inline function κuᶜᶜᶜ(i, j, k, grid, closure, surface_buoyancy_flux, S², N², w★)
    ℓᵘ = momentum_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, S², N², w★)
    κᵘ = @inbounds ℓᵘ * w★[i, j, k]
    κᵘ_max = closure.maximum_viscosity
    return min(κᵘ, κᵘ_max)
end

@inline function κcᶜᶜᶠ(i, j, k, grid, closure, surface_buoyancy_flux, S², N², w★)
    w★ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, w★)
    ℓᶜ = tracer_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, surface_buoyancy_flux, S², N², w★)
    κᶜ = ℓᶜ * w★ᶜᶜᶠ
    κᶜ_max = closure.maximum_tracer_diffusivity
    return min(κᶜ, κᶜ_max)
end

@inline function κcᶜᶜᶜ(i, j, k, grid, closure, surface_buoyancy_flux, S², N², w★)
    ℓᶜ = tracer_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, surface_buoyancy_flux, S², N², w★)
    κᶜ = @inbounds ℓᶜ * w★[i, j, k]
    κᶜ_max = closure.maximum_tracer_diffusivity
    return min(κᶜ, κᶜ_max)
end

@inline function κeᶜᶜᶠ(i, j, k, grid, closure, surface_buoyancy_flux, S², N², w★)
    w★ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, w★)
    ℓᵉ = TKE_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, surface_buoyancy_flux, S², N², w★)
    κᵉ = ℓᵉ * w★ᶜᶜᶠ
    κᵉ_max = closure.maximum_tke_diffusivity
    return min(κᵉ, κᵉ_max)
end

@inline viscosity(::FlavorOfCATKE, diffusivities) = diffusivities.κᵘ
@inline diffusivity(::FlavorOfCATKE, diffusivities, ::Val{id}) where id = diffusivities._tupled_tracer_diffusivities[id]
    
#####
##### Show
#####

function Base.summary(closure::CATKEVD)
    TD = nameof(typeof(time_discretization(closure)))
    return string("CATKEVerticalDiffusivity{$TD}")
end

function Base.show(io::IO, closure::FlavorOfCATKE)
    print(io, summary(closure))
    print(io, '\n')
    print(io, "├── maximum_tracer_diffusivity: ", prettysummary(closure.maximum_tracer_diffusivity), '\n',
              "├── maximum_tke_diffusivity: ", prettysummary(closure.maximum_tke_diffusivity), '\n',
              "├── maximum_viscosity: ", prettysummary(closure.maximum_viscosity), '\n',
              "├── minimum_turbulent_kinetic_energy: ", prettysummary(closure.minimum_turbulent_kinetic_energy), '\n',
              "├── negative_turbulent_kinetic_energy_damping_time_scale: ", prettysummary(closure.negative_turbulent_kinetic_energy_damping_time_scale), '\n',
              "├── minimum_convective_buoyancy_flux: ", prettysummary(closure.minimum_convective_buoyancy_flux), '\n',
              "├── mixing_length: ", prettysummary(closure.mixing_length), '\n',
              "│   ├── Cˢ:   ", prettysummary(closure.mixing_length.Cˢ), '\n',
              "│   ├── Cᵇ:   ", prettysummary(closure.mixing_length.Cᵇ), '\n',
              "│   ├── Cᶜc:  ", prettysummary(closure.mixing_length.Cᶜc), '\n',
              "│   ├── Cᶜe:  ", prettysummary(closure.mixing_length.Cᶜe), '\n',
              "│   ├── Cᵉc:  ", prettysummary(closure.mixing_length.Cᵉc), '\n',
              "│   ├── Cᵉe:  ", prettysummary(closure.mixing_length.Cᵉe), '\n',
              "│   ├── Cˡᵒu: ", prettysummary(closure.mixing_length.Cˡᵒu), '\n',
              "│   ├── Cˡᵒc: ", prettysummary(closure.mixing_length.Cˡᵒc), '\n',
              "│   ├── Cˡᵒe: ", prettysummary(closure.mixing_length.Cˡᵒe), '\n',
              "│   ├── Cʰⁱu: ", prettysummary(closure.mixing_length.Cʰⁱu), '\n',
              "│   ├── Cʰⁱc: ", prettysummary(closure.mixing_length.Cʰⁱc), '\n',
              "│   ├── Cʰⁱe: ", prettysummary(closure.mixing_length.Cʰⁱe), '\n',
              "│   ├── CRiᵟ: ", prettysummary(closure.mixing_length.CRiᵟ), '\n',
              "│   └── CRi⁰: ", prettysummary(closure.mixing_length.CRi⁰), '\n',
              "└── turbulent_kinetic_energy_equation: ", prettysummary(closure.turbulent_kinetic_energy_equation), '\n',
              "    ├── CˡᵒD: ", prettysummary(closure.turbulent_kinetic_energy_equation.CˡᵒD),  '\n',
              "    ├── CʰⁱD: ", prettysummary(closure.turbulent_kinetic_energy_equation.CʰⁱD),  '\n',
              "    ├── CᶜD:  ", prettysummary(closure.turbulent_kinetic_energy_equation.CᶜD),  '\n',
              "    ├── CᵉD:  ", prettysummary(closure.turbulent_kinetic_energy_equation.CᵉD),  '\n',
              "    ├── Cᵂu★: ", prettysummary(closure.turbulent_kinetic_energy_equation.Cᵂu★), '\n',
              "    ├── CᵂwΔ: ", prettysummary(closure.turbulent_kinetic_energy_equation.CᵂwΔ), '\n',
              "    └── Cᵂϵ:  ", prettysummary(closure.turbulent_kinetic_energy_equation.Cᵂϵ))
end

end # module
