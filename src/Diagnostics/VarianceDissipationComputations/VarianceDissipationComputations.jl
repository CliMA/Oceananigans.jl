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
                                       HorizontalFormulation

using Oceananigans.Advection: _advective_tracer_flux_x, 
                              _advective_tracer_flux_y, 
                              _advective_tracer_flux_z

using Oceananigans.Operators: volume
using KernelAbstractions: @kernel, @index
using Oceananigans: AbstractDiagnostic

import Oceananigans: run_diagnostic!

struct VarianceDissipation{P, K, A, D, S, G, T} <: AbstractDiagnostic
    schedule :: T
    advective_production :: P
    diffusive_production :: K
    advective_fluxes :: A
    diffusive_fluxes :: D
    previous_state :: S
    gradient_squared :: G
end

include("dissipation_utils.jl")

function VarianceDissipation(model; tracers=propertynames(model.tracers))
        
    if !(model.timestepper isa QuasiAdamsBashforth2TimeStepper)
        throw(ArgumentError("DissipationComputation requires a QuasiAdamsBashforth2TimeStepper"))
    end

    tracers = tupleit(tracers)
    diffusivities = model.diffusivity_fields
    closure       = model.closure
    grid          = model.grid

    P    = NamedTuple{tracers}(tracer_fluxes(grid) for tracer in tracers)
    K    = NamedTuple{tracers}(tracer_closure_dissipation(grid, diffusivities, closure, id) for id in eachindex(tracers))
    Vⁿ   = NamedTuple{tracers}(tracer_closure_dissipation(grid, diffusivities, closure, id) for id in eachindex(tracers))
    Vⁿ⁻¹ = NamedTuple{tracers}(tracer_closure_dissipation(grid, diffusivities, closure, id) for id in eachindex(tracers))    

    K    = NamedTuple{tracers}(tracer_fluxes(grid) for tracer in tracers)
    Fⁿ   = NamedTuple{tracers}(tracer_fluxes(grid) for tracer in tracers)
    Fⁿ⁻¹ = NamedTuple{tracers}(tracer_fluxes(grid) for tracer in tracers)
    
    Uⁿ⁻¹ = VelocityFields(grid)
    Uⁿ   = VelocityFields(grid)
    
    cⁿ⁻¹ = NamedTuple{tracers}(CenterField(grid) for tracer in tracers)

    previous_state   = merge(cⁿ⁻¹, (; Uⁿ⁻¹, Uⁿ))
    advective_fluxes = (; Fⁿ, Fⁿ⁻¹)
    diffusive_fluxes = (; Vⁿ, Vⁿ⁻¹)

    gradients = deepcopy(P)

    # Hardcode to 1 for the moment because it works only 
    # if called every time step
    schedule = IterationInterval(1)

    return VarianceDissipation(schedule, P, K, advective_fluxes, diffusive_fluxes, previous_state, gradients)
end

run_diagnostic!(ϵ::VarianceDissipation, model) = ϵ(model)

function (ϵ::VarianceDissipation)(model)

    # We first assemble values for Pⁿ⁻¹
    assemble_dissipation!(model, ϵ)

    # Then we update the fluxes to be used in the next time step
    update_fluxes!(model, ϵ)

    return nothing
end

const f = Face()
const c = Center()

include("update_fluxes.jl")
include("advective_dissipation.jl")
include("assemble_dissipation.jl")
include("flatten_dissipation_fields.jl")

end
