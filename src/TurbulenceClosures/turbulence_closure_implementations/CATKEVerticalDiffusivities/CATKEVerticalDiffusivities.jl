module CATKEVerticalDiffusivities

using Adapt
using KernelAbstractions: @kernel, @index

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Utils
using Oceananigans.Fields
using Oceananigans.Fields: ZeroField
using Oceananigans.BoundaryConditions: default_prognostic_bc, DefaultBoundaryCondition
using Oceananigans.BoundaryConditions: BoundaryCondition, FieldBoundaryConditions, DiscreteBoundaryFunction, FluxBoundaryCondition
using Oceananigans.BuoyancyModels: ∂z_b, top_buoyancy_flux
using Oceananigans.Operators: ℑzᵃᵃᶜ

using Oceananigans.TurbulenceClosures:
    getclosure,
    time_discretization,
    AbstractTurbulenceClosure,
    AbstractScalarDiffusivity,
    ConvectiveAdjustmentVerticalDiffusivity,
    ExplicitTimeDiscretization,
    VerticallyImplicitTimeDiscretization,
    ThreeDimensionalFormulation, 
    VerticalFormulation

import Oceananigans.BoundaryConditions: getbc
import Oceananigans.Utils: with_tracers
import Oceananigans.TurbulenceClosures:
    validate_closure,
    add_closure_specific_boundary_conditions,
    calculate_diffusivities!,
    DiffusivityFields,
    implicit_linear_coefficient,
    viscosity,
    diffusivity,
    diffusive_flux_x,
    diffusive_flux_y,
    diffusive_flux_z

function hydrostatic_turbulent_kinetic_energy_tendency end

struct CATKEVerticalDiffusivity{TD, CL, TKE} <: AbstractScalarDiffusivity{TD, VerticalFormulation}
    mixing_length :: CL
    turbulent_kinetic_energy_equation :: TKE
end

function CATKEVerticalDiffusivity{TD}(mixing_length :: CL,
                                      turbulent_kinetic_energy_equation :: TKE) where {TD, CL, TKE}

    return CATKEVerticalDiffusivity{TD, CL, TKE}(mixing_length,
                                                 turbulent_kinetic_energy_equation)
end

const CATKEVD{TD} = CATKEVerticalDiffusivity{TD} where TD

# Support for "ManyIndependentColumnMode"
const CATKEVDArray{TD} = AbstractArray{<:CATKEVD{TD}} where TD
const FlavorOfCATKE{TD} = Union{CATKEVD{TD}, CATKEVDArray{TD}} where TD

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

# Two CATKEs?!
catke_first(catke1::FlavorOfCATKE, catke2::FlavorOfCATKE) = error("Can't have two CATKEs in one closure tuple.")

include("mixing_length.jl")
include("turbulent_kinetic_energy_equation.jl")

for S in (:MixingLength, :TurbulentKineticEnergyEquation)
    @eval @inline convert_eltype(::Type{FT}, s::$S) where FT = $S{FT}(; Dict(p => getproperty(s, p) for p in propertynames(s))...)
    @eval @inline convert_eltype(::Type{FT}, s::$S{FT}) where FT = s
end

"""
    CATKEVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(), FT=Float64;
                             mixing_length = MixingLength{FT}(),
                             turbulent_kinetic_energy_equation = TurbulentKineticEnergyEquation{FT}(),
                             warning = true)

Returns the `CATKEVerticalDiffusivity` turbulence closure for vertical mixing by
small-scale ocean turbulence based on the prognostic evolution of subgrid
Turbulent Kinetic Energy (TKE).
"""
CATKEVerticalDiffusivity(FT::DataType; kw...) = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

function CATKEVerticalDiffusivity(time_discretization::TD = VerticallyImplicitTimeDiscretization(), FT=Float64;
                                  mixing_length = MixingLength{FT}(),
                                  turbulent_kinetic_energy_equation = TurbulentKineticEnergyEquation{FT}(),
                                  warning = true) where TD

    if warning
        @warn "CATKEVerticalDiffusivity is an experimental turbulence closure that \n" *
              "is unvalidated and whose default parameters are not calibrated for \n" * 
              "realistic ocean conditions or for use in a three-dimensional \n" *
              "simulation. Use with caution and report bugs and problems with physics \n" *
              "to https://github.com/CliMA/Oceananigans.jl/issues."
    end

    mixing_length = convert_eltype(FT, mixing_length)
    turbulent_kinetic_energy_equation = convert_eltype(FT, turbulent_kinetic_energy_equation)

    return CATKEVerticalDiffusivity{TD}(mixing_length, turbulent_kinetic_energy_equation)
end

#####
##### Diffusivities and diffusivity fields utilities
#####

function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfCATKE)

    default_diffusivity_bcs = (Kᵘ = FieldBoundaryConditions(grid, (Center, Center, Face)),
                               Kᶜ = FieldBoundaryConditions(grid, (Center, Center, Face)),
                               Kᵉ = FieldBoundaryConditions(grid, (Center, Center, Face)))

    bcs = merge(default_diffusivity_bcs, bcs)

    Kᵘ = CenterField(grid, boundary_conditions=bcs.Kᵘ)
    Kᶜ = CenterField(grid, boundary_conditions=bcs.Kᶜ)
    Kᵉ = CenterField(grid, boundary_conditions=bcs.Kᵉ)
    Lᵉ = CenterField(grid) #, boundary_conditions=nothing)

    # Secret tuple for getting tracer diffusivities with tuple[tracer_index]
    _tupled_tracer_diffusivities         = NamedTuple(name => name === :e ? Kᵉ : Kᶜ          for name in tracer_names)
    _tupled_implicit_linear_coefficients = NamedTuple(name => name === :e ? Lᵉ : ZeroField() for name in tracer_names)

    return (; Kᵘ, Kᶜ, Kᵉ, Lᵉ, _tupled_tracer_diffusivities, _tupled_implicit_linear_coefficients)
end        

@inline viscosity_location(::FlavorOfCATKE) = (Center(), Center(), Face())
@inline diffusivity_location(::FlavorOfCATKE) = (Center(), Center(), Face())

@inline clip(x) = max(zero(x), x)

function calculate_diffusivities!(diffusivities, closure::FlavorOfCATKE, model)

    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = model.tracers
    buoyancy = model.buoyancy
    clock = model.clock
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

    event = launch!(arch, grid, :xyz,
                    calculate_CATKE_diffusivities!,
                    diffusivities, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

@kernel function calculate_CATKE_diffusivities!(diffusivities, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy, args...)
    i, j, k, = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)

    @inbounds begin
        diffusivities.Kᵘ[i, j, k] = Kuᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, args...)
        diffusivities.Kᶜ[i, j, k] = Kcᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, args...)
        diffusivities.Kᵉ[i, j, k] = Keᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, args...)

        # "Patankar trick" for buoyancy production (cf Patankar 1980 or Burchard et al. 2003)
        # If buoyancy flux is a _sink_ of TKE, we treat it implicitly.
        Qᵇ = buoyancy_flux(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, diffusivities)
        eⁱʲᵏ = @inbounds tracers.e[i, j, k]
        Qᵇ_e = ifelse(sign(Qᵇ) * sign(eⁱʲᵏ) < 0, Qᵇ / eⁱʲᵏ, zero(grid))
        
        diffusivities.Lᵉ[i, j, k] = - Qᵇ_e + implicit_dissipation_coefficient(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, args...)
    end
end

@inline function implicit_linear_coefficient(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, K, ::Val{id}, args...) where id
    L = K._tupled_implicit_linear_coefficients[id]
    return @inbounds L[i, j, k]
end

@inline turbulent_velocity(i, j, k, grid, e) = @inbounds sqrt(clip(e[i, j, k]))
@inline is_stableᶜᶜᶠ(i, j, k, grid, tracers, buoyancy) = ∂z_b(i, j, k, grid, buoyancy, tracers) >= 0

@inline function Kuᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    u★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocity, tracers.e)
    ℓu = momentum_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    return ℓu * u★
end

@inline function Kcᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    u★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocity, tracers.e)
    ℓc = tracer_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    return ℓc * u★
end

@inline function Keᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    u★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocity, tracers.e)
    ℓe = TKE_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    return ℓe * u★
end

@inline viscosity(::FlavorOfCATKE, diffusivities) = diffusivities.Kᵘ
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
    show(io, closure.mixing_length)
    print(io, '\n')
    show(io, closure.turbulent_kinetic_energy_equation)
end

end # module
