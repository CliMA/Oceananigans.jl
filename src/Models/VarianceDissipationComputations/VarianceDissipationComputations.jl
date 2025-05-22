module VarianceDissipationComputations

export VarianceDissipation, flatten_dissipation_fields

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

using Oceananigans: UpdateStateCallsite
using Oceananigans.Operators: volume
using Oceananigans.Utils: IterationInterval, ConsecutiveIterations
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

function c_grid_vector(grid)
    x = XFaceField(grid)
    y = YFaceField(grid)
    z = ZFaceField(grid)
    return (; x, y, z)
end

"""
    VarianceDissipation(tracer_name, grid;
                        Uⁿ⁻¹ = VelocityFields(grid),
                        Uⁿ   = VelocityFields(grid))

Construct a `VarianceDissipation` object for a tracer called `tracer_name` that lives on a `grid`.
This function computes the variance dissipation diagnostics for the specified tracer in the model.
These include the numerical dissipation implicit to the advection scheme and the explicit
dissipation associated to closures.

This diagnostic is especially useful for models that use a dissipative advection scheme
like [`WENO`](@ref) or [`UpwindBiased`](@ref).

Arguments
=========

- `tracer_name`: The name of the tracer for which variance dissipation is computed. This should be a `Symbol`.
                 When calling `ϵ::VarianceDissipation` on the model, this name is used to identify the tracer in the model's state.
- `grid`: The grid on which the tracer is defined.

Keyword Arguments
=================

- `Uⁿ⁻¹`: The velocity field at the previous time step. Default: `VelocityFields(grid)`.
- `Uⁿ`: The velocity field at the current time step. Default: `VelocityFields(grid)`.

!!! compat "Time stepper compatibility"
    At the moment, the variance dissipation diagnostic is not supported for a [`RungeKutta3TimeStepper`](@ref Oceananigans.TimeSteppers.RungeKutta3TimeStepper).
"""
function VarianceDissipation(tracer_name, grid;
                             Uⁿ⁻¹ = VelocityFields(grid),
                             Uⁿ   = VelocityFields(grid))

    P    = c_grid_vector(grid)
    K    = c_grid_vector(grid)
    Vⁿ   = c_grid_vector(grid)
    Vⁿ⁻¹ = c_grid_vector(grid)
    Fⁿ   = c_grid_vector(grid)
    Fⁿ⁻¹ = c_grid_vector(grid)
    cⁿ⁻¹ = CenterField(grid)

    previous_state   = (; cⁿ⁻¹, Uⁿ⁻¹, Uⁿ)
    advective_fluxes = (; Fⁿ, Fⁿ⁻¹)
    diffusive_fluxes = (; Vⁿ, Vⁿ⁻¹)

    gradients = deepcopy(P)

    return VarianceDissipation(P, K, advective_fluxes, diffusive_fluxes, previous_state, gradients, tracer_name)
end

function (ϵ::VarianceDissipation)(model)

    # Check if the timestepper is supported
    if model.timestepper isa RungeKutta3TimeStepper
        throw(ArgumentError("`VarianceDissipation` using a `RungeKutta3TimeStepper` are not supported."))
    end

    # Check if the model has a velocity field
    if !hasproperty(model, :velocities)
        throw(ArgumentError("Model must have a velocity field."))
    end

    # Check if the model has tracers
    if !hasproperty(model.tracers, ϵ.tracer_name)
        throw(ArgumentError("Model must have a tracer called $tracer_name."))
    end

    # First we compute the dissipation from previously computed fluxes
    compute_dissipation!(ϵ, model, ϵ.tracer_name)

    # Then we update the fluxes to be used in the next time step
    cache_fluxes!(ϵ, model, ϵ.tracer_name)

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
