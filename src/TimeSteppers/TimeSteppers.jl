module TimeSteppers

export 
    AdamsBashforthTimeStepper,
    time_step!, 
    compute_w_from_continuity!

using Oceananigans: device

using GPUifyLoops: @launch, @loop, @unroll

import Oceananigans: TimeStepper

using Oceananigans: AbstractGrid, Model, Tendencies, tracernames, 
                    @hascuda, CPU, GPU, launch_config, datatuples, datatuple,
                    @loop_xyz,

                    buoyancy_perturbation,
                    x_f_cross_U, y_f_cross_U, z_f_cross_U,

                    fill_halo_regions!, apply_z_bcs!, solve_poisson_3d!, PoissonBCs, PPN, PNN,

                    run_diagnostic, write_output, time_to_run

@hascuda using CUDAnative, CUDAdrv, CuArrays

using ..Operators

using ..TurbulenceClosures: ∂ⱼ_2ν_Σ₁ⱼ, ∂ⱼ_2ν_Σ₂ⱼ, ∂ⱼ_2ν_Σ₃ⱼ, ∇_κ_∇c,
                            calculate_diffusivities!, ▶z_aaf

const Tx = 16 # CUDA threads per x-block
const Ty = 16 # CUDA threads per y-block

"""
    TimeStepper(name, args...)

Returns a timestepper with name `name`, instantiated with `args...`.

Example
=======

julia> stepper = TimeStepper(:AdamsBashforth, Float64, CPU(), grid, tracernames)
"""
function TimeStepper(name::Symbol, args...)
    fullname = Symbol(name, :TimeStepper)
    return eval(Expr(:call, fullname, args...))
end

# Fallback
TimeStepper(stepper, args...) = stepper

"""Returns the arguments passed to boundary conditions functions."""
boundary_condition_function_arguments(model) = 
    (model.clock.time, model.clock.iteration, datatuple(model.velocities), 
     datatuple(model.tracers), model.parameters)

####
#### Time-stepping kernels/function that are independent of the TimeStepper
####

"""
    time_step!(model; Nt, Δt, kwargs...)

Step forward `model` `Nt` time steps with step size `Δt`.

The kwargs are passed to the `time_step!` function specific to `model.timestepper`.
"""
time_step!(model; Nt, Δt, kwargs...) = time_step!(model, Nt, Δt; kwargs...)

function time_step!(model, Nt, Δt; kwargs...)

    if model.clock.iteration == 0
        [ run_diagnostic(model, diag) for diag in values(model.diagnostics) ]
        [ write_output(model, out)    for out  in values(model.output_writers) ]
    end

    for n in 1:Nt
        time_step!(model, Δt; kwargs...)

        [ time_to_run(model.clock, diag) && run_diagnostic(model, diag) for diag in values(model.diagnostics) ]
        [ time_to_run(model.clock, out) && write_output(model, out) for out in values(model.output_writers) ]
    end

    return nothing
end

"""
    time_step_precomputations!(diffusivities, pressures, velocities, tracers, model)

Perform precomputations necessary for an explicit timestep or substep.
"""
function time_step_precomputations!(diffusivities, pressures, velocities, tracers, model)

    fill_halo_regions!(merge(velocities, tracers), model.boundary_conditions.solution, model.architecture, 
                       model.grid, boundary_condition_function_arguments(model)...)

    calculate_diffusivities!(diffusivities, model.architecture, model.grid, model.closure, model.buoyancy, 
                             velocities, tracers)

    # Diffusivities share bcs with pressure:
    fill_halo_regions!(diffusivities, model.boundary_conditions.pressure, model.architecture, model.grid) 

    @launch(device(model.architecture), config=launch_config(model.grid, 2), 
            update_hydrostatic_pressure!(pressures.pHY′, model.grid, model.buoyancy, tracers))

    fill_halo_regions!(pressures.pHY′, model.boundary_conditions.pressure, model.architecture, model.grid)

    return nothing
end

"""
    calculate_tendencies!(diffusivities, pressures, velocities, tracers, model)

Calculate the interior and boundary contributions to tendency terms without the 
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(tendencies, velocities, tracers, pressures, diffusivities, model)

    calculate_interior_source_terms!(tendencies, model.architecture, model.grid, model.coriolis, model.buoyancy,
                                     model.closure, velocities, tracers, pressures.pHY′, diffusivities,
                                     model.forcing, model.parameters, model.clock.time)

    calculate_boundary_source_terms!(tendencies, model.boundary_conditions.solution, model.architecture, 
                                     model.grid, boundary_condition_function_arguments(model)...)
                                     
    return nothing
end

"""
    calculate_pressure_correction!(nonhydrostatic_pressure, Δt, tendencies, velocities, model)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function calculate_pressure_correction!(nonhydrostatic_pressure, Δt, tendencies, velocities, model)
    velocity_tendencies = (u=tendencies.u, v=tendencies.v, w=tendencies.w)

    velocity_tendency_boundary_conditions = (u=model.boundary_conditions.tendency.u, 
                                             v=model.boundary_conditions.tendency.v, 
                                             w=model.boundary_conditions.tendency.w)

    fill_halo_regions!(velocity_tendencies, velocity_tendency_boundary_conditions, model.architecture, model.grid)

    @launch(device(model.architecture), config=launch_config(model.grid, 3), 
            calculate_poisson_right_hand_side!(model.poisson_solver.storage, model.architecture, model.grid,
                                               model.poisson_solver.bcs, velocities, tendencies, Δt))

    solve_for_pressure!(nonhydrostatic_pressure, model.architecture, model.grid, model.poisson_solver, 
                        model.poisson_solver.storage)

    fill_halo_regions!(nonhydrostatic_pressure, model.boundary_conditions.pressure, model.architecture, model.grid)

    return nothing
end

calculate_pressure_correction!(::Nothing, args...) = nothing

"""
    calculate_explicit_substep!(tendencies, velocities, tracers, pressures, diffusivities, model)

Calculate the initial and explicit substep of the two-step fractional step method with pressure correction.
"""
function calculate_explicit_substep!(tendencies, velocities, tracers, pressures, diffusivities, model)
    time_step_precomputations!(diffusivities, pressures, velocities, tracers, model)
    calculate_tendencies!(tendencies, velocities, tracers, pressures, diffusivities, model)
    return nothing
end

"""
    complete_pressure_correction_step!(velocities, Δt, tracers, pressures, tendencies, model)

After calculating the pressure correction, complete the pressure correction step by updating
the velocity and tracer fields.
"""
function complete_pressure_correction_step!(velocities, Δt, tracers, pressures, tendencies, model)
    update_solution!(velocities, tracers, model.architecture, model.grid, Δt, tendencies, pressures.pNHS)

    velocity_boundary_conditions = (u=model.boundary_conditions.solution.u, 
                                    v=model.boundary_conditions.solution.v, 
                                    w=model.boundary_conditions.solution.w)

    # Recompute vertical velocity w from continuity equation to ensure incompressibility
    fill_halo_regions!(velocities, velocity_boundary_conditions, model.architecture, model.grid, 
                       boundary_condition_function_arguments(model)...)

    compute_w_from_continuity!(model)

    return nothing
end

include("kernels.jl")
include("adams_bashforth.jl")

end # module
