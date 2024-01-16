using CUDA

using Oceananigans.Fields: AbstractField, compute_at!, ZeroField
using Oceananigans.ImmersedBoundaries: mask_immersed!
using Oceananigans.Models.LagrangianParticleTracking: LagrangianParticles

# Needed to support `fetch_output` with `model::Nothing`.
time(model) = model.clock.time
time(::Nothing) = nothing

# Default fetch_output with mask_immersed = nothing
fetch_output(output, model) = fetch_output(output, model, nothing)

function fetch_output(output, model, mask_immersed)
    fetched = output(model)
    if fetched isa Field
        !isnothing(mask_immersed) && mask_immersed!(fetched, mask_immersed)
        return parent(fetched)
    end
    return fetched
end

function fetch_output(field::AbstractField, model, mask_immersed)
    compute_at!(field, time(model))
    !isnothing(mask_immersed) && mask_immersed!(field, mask_immersed)
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
    fetched = fetch_output(output, model, writer.mask_immersed)
    return convert_output(fetched, writer)
end

fetch_and_convert_output(output::ZeroField, model, writer) = zero(eltype(output))

