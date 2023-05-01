using CUDA

using Oceananigans.Fields: AbstractField, compute_at!
using Oceananigans.ImmersedBoundaries: mask_immersed!
using Oceananigans.LagrangianParticleTracking: LagrangianParticles

# Needed to support `fetch_output` with `model::Nothing`.
time(model) = model.clock.time
time(::Nothing) = nothing

# Default fetch_output with mask_value=nothing
fetch_output(output, model) = fetch_output(output, model, nothing)

function fetch_output(output, model, mask_value)
    fetched = output(model)
    if fetched isa Field
        !isnothing(mask_value) && mask_immersed!(fetched, mask_value)
        return parent(fetched)
    end
    return fetched
end

function fetch_output(field::AbstractField, model, mask_value)
    compute_at!(field, time(model))
    !isnothing(mask_value) && mask_immersed!(field, mask_value)
    return parent(field)
end

function fetch_output(lagrangian_particles::LagrangianParticles, model)
    particle_properties = lagrangian_particles.properties
    names = propertynames(particle_properties)
    return NamedTuple{names}([getproperty(particle_properties, name) for name in names])
end

convert_output(output, writer) = output

function convert_output(output::AbstractArray, writer)
    if architecture(output) isa GPU
        output_array = writer.array_type(undef, size(output)...)
        copyto!(output_array, output)
    else
        output_array = convert(writer.array_type, output)
    end

    return output_array
end

# Need to broadcast manually because of https://github.com/JuliaLang/julia/issues/30836
convert_output(outputs::NamedTuple, writer) =
    NamedTuple(name => convert_output(outputs[name], writer) for name in keys(outputs))

function fetch_and_convert_output(output, model, writer)
    fetched = fetch_output(output, model, writer.mask_value)
    return convert_output(fetched, writer)
end
