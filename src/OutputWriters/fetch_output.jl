using CUDA

using Oceananigans.Fields: AbstractField, compute_at!
using Oceananigans.LagrangianParticleTracking: LagrangianParticles

fetch_output(output, model, field_slicer) = output(model)

# Needed to support `fetch_output` with `model::Nothing`.
time(model) = model.clock.time
time(::Nothing) = nothing

function fetch_output(field::AbstractField, model, field_slicer)
    compute_at!(field, time(model))
    return slice_parent(field_slicer, field)
end

function fetch_output(lagrangian_particles::LagrangianParticles, model, field_slicer)
    particles = lagrangian_particles.particles
    names = propertynames(particles)
    return NamedTuple{names}([getproperty(particles, name) for name in names])
end

convert_output(output, writer) = output
convert_output(output::AbstractArray, writer) = CUDA.@allowscalar writer.array_type(output)

# Need to broadcast manually because of https://github.com/JuliaLang/julia/issues/30836
convert_output(outputs::NamedTuple, writer) =
    NamedTuple{keys(outputs)}(writer.array_type.(values(outputs)))

fetch_and_convert_output(output, model, writer) =
    convert_output(fetch_output(output, model, writer.field_slicer), writer)
