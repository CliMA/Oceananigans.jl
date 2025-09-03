using Ocenanigans.Utils
import Oceananigans.Fields: set!

#####
##### set! for checkpointer filepaths and `OceananigansModels`
#####

"""
    set!(model::OceananigansModels, filepath::AbstractString)

Set data in `model.velocities`, `model.tracers`, `model.timestepper.Gⁿ`, and
`model.timestepper.G⁻` to checkpointed data stored at `filepath`.
"""
function set!(model::OceananigansModels, filepath::AbstractString)
    addr = checkpointer_address(model)

    jldopen(filepath, "r") do file

        # Validate the grid
        checkpointed_grid = file["$addr/grid"]

        model.grid == checkpointed_grid ||
            @warn "The grid associated with $filepath and model.grid are not the same!"

        model_fields = prognostic_fields(model)

        for name in keys(model_fields)
            if string(name) ∈ keys(file[addr]) # Test if variable exists in checkpoint.
                model_field = model_fields[name]
                parent_data = on_architecture(model.architecture, file["$addr/$name/data"])
                @apply_regionally copyto!(parent(model_field), parent_data)
            else
                @warn "Field $name does not exist in checkpoint and could not be restored."
            end
        end

        set_time_stepper!(model.timestepper, model.architecture, file, model_fields, addr)

        # Try restoring particles
        if :particles ∈ keys(file[addr]) && !isnothing(model.particles)
            copyto!(model.particles.properties, file["$addr/particles"])
        else
            @warn "Particles do not exist in checkpoint and could not be restored."
        end

        checkpointed_clock = file["$addr/clock"]

        # Update model clock
        model.clock.iteration = checkpointed_clock.iteration
        model.clock.time = checkpointed_clock.time
        model.clock.last_Δt = checkpointed_clock.last_Δt
    end

    return nothing
end

function set_time_stepper_tendencies!(timestepper, arch, file, model_fields, addr)
    for name in propertynames(model_fields)
        tendency_in_model = hasproperty(timestepper.Gⁿ, name)
        tendency_in_checkpoint = string(name) ∈ keys(file["$addr/timestepper/Gⁿ"])
        if tendency_in_model && tendency_in_checkpoint
            # Tendency "n"
            parent_data = on_architecture(arch, file["$addr/timestepper/Gⁿ/$name/data"])

            tendencyⁿ_field = timestepper.Gⁿ[name]
            @apply_regionally copyto!(parent(tendencyⁿ_field), parent_data)

            # Tendency "n-1"
            parent_data = on_architecture(arch, file["$addr/timestepper/G⁻/$name/data"])

            tendency⁻_field = timestepper.G⁻[name]
            @apply_regionally copyto!(parent(tendency⁻_field), parent_data)
        elseif tendency_in_model && !tendency_in_checkpoint
            @warn "Tendencies for $name do not exist in checkpoint and could not be restored."
        end
    end

    return nothing
end

# For self-starting timesteppers like RK3 we do nothing
set_time_stepper!(timestepper, args...) = nothing

set_time_stepper!(timestepper::QuasiAdamsBashforth2TimeStepper, args...) =
    set_time_stepper_tendencies!(timestepper, args...)
