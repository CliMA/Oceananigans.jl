using CUDA

using Oceananigans.Fields: AbstractField, compute_at!
using Oceananigans.Models.LagrangianParticleTracking: LagrangianParticles

# Needed to support `fetch_output` with `model::Nothing`.
time(model) = model.clock.time
time(::Nothing) = nothing

fetch_output(output, model) = output(model)

function fetch_output(field::AbstractField, model)
    compute_at!(field, time(model))
    return parent(field)
end

function fetch_output(lagrangian_particles::LagrangianParticles, model)
    particle_properties = lagrangian_particles.properties
    names = propertynames(particle_properties)
    return NamedTuple{names}([getproperty(particle_properties, name) for name in names])
end

convert_output(output, writer) = output
convert_output(output::AbstractArray, writer) = Array{writer.type}(output)

# Need to broadcast manually because of https://github.com/JuliaLang/julia/issues/30836
convert_output(outputs::NamedTuple, writer) =
    NamedTuple(name => convert_output(outputs[name], writer) for name in keys(outputs))

function fetch_and_convert_output(output, model, writer)
    fetched = fetch_output(output, model)
    return convert_output(fetched, writer)
end
