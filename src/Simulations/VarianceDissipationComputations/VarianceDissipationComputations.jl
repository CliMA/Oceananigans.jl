module VarianceDissipationComputations
 
export VarianceDissipation, get_dissipation_fields

using Oceananigans.Grids: architecture
using Oceananigans.Utils
using Oceananigans.TimeSteppers
using Oceananigans.Fields
using Oceananigans.Fields: Field, VelocityFields
using Oceananigans.Operators
using Oceananigans.BoundaryConditions
using Oceananigans.TurbulenceClosures: viscosity,
                                       diffusivity, 
                                       ScalarDiffusivity, 
                                       ScalarBiharmonicDiffusivity,
                                       AbstractTurbulenceClosure,
                                       HorizontalFormulation,
                                       _diffusive_flux_x,
                                       _diffusive_flux_y,
                                       _diffusive_flux_z

using Oceananigans.Advection: _advective_tracer_flux_x, 
                              _advective_tracer_flux_y, 
                              _advective_tracer_flux_z

using Oceananigans.Operators: volume
using KernelAbstractions: @kernel, @index

struct VarianceDissipation{P, K, A, D, S, G} 
    advective_production :: P
    diffusive_production :: K
    advective_fluxes :: A
    diffusive_fluxes :: D
    previous_state :: S
    gradient_squared :: G
    tracer_name :: Symbol
end

function vector_field(grid)
    x = XFaceField(grid)
    y = YFaceField(grid)
    z = ZFaceField(grid)
    return (; x, y, z)
end

"""
    VarianceDissipation(model; tracers=propertynames(model.tracers))

Constructs a `VarianceDissipation` object for a given `model`. This function computes 
the variance dissipation diagnostics for the specified tracers in the model. 
These include the numerical dissipation implicit to the advection scheme and the explicit 
dissipation associated to closures. 

This diagnostic is especially useful for models that use a dissipative advection scheme
like [`WENO`](@ref) or [`UpwindBiased`](@ref)

Argument
=========

- `model`: The model for which variance dissipation is to be computed. 
  The model must use a `QuasiAdamsBashforth2TimeStepper` for its time-stepping scheme.

Keyword Argument
================

- `tracers`: An optional argument specifying the tracers for which variance dissipation 
  is computed (can be a `Symbol` or a `Tuple` of `Symbols`). All symbols must be tracers evolved
  by the `model`. Default: `propertynames(model.tracers)`.

!!! Note
    At the moment, the variance dissipation diagnostic is supported only for `QuasiAdamsBashforth2` timesteppers.
"""
function VarianceDissipation(tracer_name, grid; 
                             Uⁿ⁻¹ = VelocityFields(tracer.grid), 
                             Uⁿ   = VelocityFields(tracer.grid))
        
    P    = vector_field(grid) 
    K    = vector_field(grid)
    Vⁿ   = vector_field(grid) 
    Vⁿ⁻¹ = vector_field(grid) 
    Fⁿ   = vector_field(grid) 
    Fⁿ⁻¹ = vector_field(grid) 
    cⁿ⁻¹ = CenterField(grid)

    previous_state   = merge(cⁿ⁻¹, (; Uⁿ⁻¹, Uⁿ))
    advective_fluxes = (; Fⁿ, Fⁿ⁻¹)
    diffusive_fluxes = (; Vⁿ, Vⁿ⁻¹)

    gradients = deepcopy(P)

    return VarianceDissipation(P, K, advective_fluxes, diffusive_fluxes, previous_state, gradients, tracer_name)
end

function (ϵ::VarianceDissipation)(sim::Simulation)
    model = sim.model
    # Check if the model is using a QuasiAdamsBashforth2 time stepper
    if !(model.timestepper isa QuasiAdamsBashforth2TimeStepper)
        throw(ArgumentError("VarianceDissipation is only supported for QuasiAdamsBashforth2 time-stepping."))
    end

    # Check if the model has a velocity field
    if !hasproperty(model, :velocities)
        throw(ArgumentError("Model must have a velocity field."))
    end

    # Check if the model has tracers
    if !hasproperty(model.tracers, ϵ.tracer_name)
        throw(ArgumentError("Model must have a tracer called $tracer_name."))
    end

    compute_dissipation!(model, ϵ, ϵ.tracer_name)

    # Then we update the fluxes to be used in the next time step
    cache_fluxes!(model, ϵ, ϵ.tracer_name)
    
    return nothing
end

@inline getadvection(advection, tracer_name) = advection
@inline getadvection(advection::NamedTuple, tracer_name) = @inbounds advection[tracer_name]

const f = Face()
const c = Center()

include("update_fluxes.jl")
include("advective_dissipation.jl")
include("diffusive_dissipation.jl")
include("compute_dissipation.jl")
include("flatten_dissipation_fields.jl")

end
