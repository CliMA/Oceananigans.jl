module VarianceDissipationComputation
 
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
                              _advective_tracer_flux_z,
                              horizontal_advection_U, 
                              horizontal_advection_V 

using Oceananigans.Operators: volume
using KernelAbstractions: @kernel, @index

struct VarianceDissipation{P, K, A, D, S, G}
    advective_production :: P
    diffusive_production :: K
    advective_fluxes :: A
    diffusive_fluxes :: D
    previous_state :: S
    gradient_squared :: G
end

include("dissipation_utils.jl")

function VarianceDissipation(model; 
                             tracers = propertynames(model.tracers), 
                             include_vorticity = true)
        
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
    
    cⁿ⁻¹ =  NamedTuple{tracers}(CenterField(grid) for tracer in tracers)

    if include_vorticity
        Fζⁿ   = vorticity_fluxes(grid)
        Fζⁿ⁻¹ = vorticity_fluxes(grid)
        Pζ    = vorticity_fluxes(grid)
        ζⁿ⁻¹  = Field{Face, Face, Center}(grid)

        P    = merge(P,    (; ζ = Pζ))
        Fⁿ   = merge(Fⁿ,   (; ζ = Fζⁿ))
        Fⁿ⁻¹ = merge(Fⁿ⁻¹, (; ζ = Fζⁿ⁻¹))
        cⁿ⁻¹ = merge(cⁿ⁻¹, (; ζ = ζⁿ⁻¹))

        Kζ    = enstrophy_closure_dissipation(grid, diffusivities, closure)
        Vζⁿ   = enstrophy_closure_dissipation(grid, diffusivities, closure)
        Vζⁿ⁻¹ = enstrophy_closure_dissipation(grid, diffusivities, closure)
            
        K    = merge(K,    (; ζ = Kζ))
        Vⁿ   = merge(Vⁿ,   (; ζ = Vζⁿ))
        Vⁿ⁻¹ = merge(Vⁿ⁻¹, (; ζ = Vζⁿ⁻¹))
    end

    previous_state   = merge(cⁿ⁻¹, (; Uⁿ⁻¹, Uⁿ))
    advective_fluxes = (; Fⁿ, Fⁿ⁻¹)
    diffusive_fluxes = (; Vⁿ, Vⁿ⁻¹)

    gradients = deepcopy(P)

    return VarianceDissipation(P, K, advective_fluxes, diffusive_fluxes, previous_state, gradients)
end

# Function to call in a callback
# Note: This works only if the callback is called with an IterationInterval(1), if not the
# previous fluxes and velocities will not be correct
# TODO: make sure that the correct velocities and fluxes are used even if 
# the callback is not called with an IterationInterval(1)
function (ϵ::VarianceDissipation)(simulation)

    # We first assemble values for Pⁿ⁻¹
    assemble_dissipation!(simulation, ϵ)

    # Then we update the fluxes to be used in the next time step
    update_fluxes!(simulation, ϵ)

    return nothing
end

include("get_dissipation_fields.jl")
include("update_fluxes.jl")
include("advective_fluxes.jl")
include("diffusive_fluxes.jl")
include("assemble_dissipation.jl")

end
