using Oceananigans.Utils: prettytime, prettykeys
using Oceananigans.TurbulenceClosures: closure_summary

function Base.summary(model::NonhydrostaticModel)
    A = Base.summary(architecture(model.grid))
    G = nameof(typeof(model.grid))
    return string("NonhydrostaticModel{$A, $G}",
                  "(time = ", prettytime(model.clock.time), ", iteration = ", model.clock.iteration, ")")
end

function Base.show(io::IO, model::NonhydrostaticModel)
    TS = nameof(typeof(model.timestepper))
    tracernames = prettykeys(model.tracers)

    print(io, summary(model), "\n",
        "├── grid: ", summary(model.grid), "\n",
        "├── timestepper: ", TS, "\n")

    if model.advection !== nothing
        print(io, "├── advection scheme: ", "\n")
        names = keys(model.advection)
        for name in names[1:end-1]
            print(io, "│   ├── " * string(name) * ": " * summary(model.advection[name]), "\n")
        end
        name = names[end]
        print(io, "│   └── " * string(name) * ": " * summary(model.advection[name]), "\n")
    end

    print(io,
        "├── tracers: ", tracernames, "\n",
        "├── closure: ", closure_summary(model.closure), "\n",
        "├── buoyancy: ", summary(model.buoyancy), "\n")

    if isnothing(model.particles)
        print(io, "└── coriolis: ", summary(model.coriolis))
    else
        particles = model.particles.properties
        properties = propertynames(particles)
        print(io, "├── coriolis: ", summary(model.coriolis), "\n")
        print(io, "└── particles: ", summary(model.particles))
    end
end
