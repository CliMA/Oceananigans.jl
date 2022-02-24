using Oceananigans.Utils: prettytime, ordered_dict_show
using Oceananigans.TurbulenceClosures: closure_summary

function Base.summary(model::NonhydrostaticModel)
    A = nameof(typeof(architecture(model.grid)))
    G = nameof(typeof(model.grid))
    return string("NonhydrostaticModel{$A, $G}",
                  "(time = ", prettytime(model.clock.time), ", iteration = ", model.clock.iteration, ")")
end

function Base.show(io::IO, model::NonhydrostaticModel)
    TS = nameof(typeof(model.timestepper))

    tracernames = keys(model.tracers)
    if length(tracernames) == 1
        tracerstr = string(first(tracernames))
    else
        tracerstr = string("(", (string(n, ", ") for n in tracernames[1:end-1])..., last(tracernames), ")")
    end

    print(io, summary(model), '\n',
        "├── grid: ", summary(model.grid), '\n',
        "├── timestepper: ", TS, '\n',
        "├── tracers: ", tracerstr, '\n',
        "├── closure: ", closure_summary(model.closure), '\n',
        "├── buoyancy: ", summary(model.buoyancy), '\n')

    if isnothing(model.particles)
        print(io, "└── coriolis: ", summary(model.coriolis))
    else
        particles = model.particles.properties
        properties = propertynames(particles)
        print(io, "├── coriolis: ", summary(model.coriolis), '\n')
        print(io, "└── particles: $(length(particles)) Lagrangian particles with $(length(properties)) properties: $properties")
    end
end

