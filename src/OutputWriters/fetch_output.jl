using CUDA

using Oceananigans.Fields: AbstractField, compute!
using Oceananigans.LagrangianParticleTracking: LagrangianParticles

fetch_output(output, model, field_slicer) = output(model)

# Needed to support `fetch_output` with `model::Nothing`.
time(model) = model.clock.time
time(::Nothing) = nothing

function fetch_output(field::AbstractField, model, field_slicer)
    compute!(field, time(model))
    return slice_parent(field_slicer, field)
end

function fetch_output(particles::LagrangianParticles, model, field_slicer)
    return (x=particles.particles.x, y=particles.particles.y, z=particles.particles.z)
end

convert_output(output, writer) = output
convert_output(output::AbstractArray, writer) = CUDA.@allowscalar writer.array_type(output)

# Need to broadcast manually because of https://github.com/JuliaLang/julia/issues/30836
convert_output(outputs::NamedTuple{(:x, :y, :z)}, writer) =
    CUDA.@allowscalar (x=writer.array_type(outputs.x), y=writer.array_type(outputs.y), z=writer.array_type(outputs.z))

fetch_and_convert_output(output, model, writer) =
    convert_output(fetch_output(output, model, writer.field_slicer), writer)
