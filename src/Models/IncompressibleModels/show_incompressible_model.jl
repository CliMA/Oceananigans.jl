using Oceananigans.Utils: prettytime, ordered_dict_show
using Oceananigans: short_show

"""Show the innards of a `Model` in the REPL."""
function Base.show(io::IO, model::IncompressibleModel{TS, C, A}) where {TS, C, A}
    print(io, "IncompressibleModel{$A, $(eltype(model.grid))}",
        "(time = $(prettytime(model.clock.time)), iteration = $(model.clock.iteration)) \n",
        "├── grid: $(short_show(model.grid))\n",
        "├── tracers: $(tracernames(model.tracers))\n",
        "├── closure: $(typeof(model.closure))\n",
        "├── buoyancy: $(typeof(model.buoyancy))\n")

    if isnothing(model.particles)
        print(io, "└── coriolis: $(typeof(model.coriolis))")
    else
        particles = model.particles.particles
        properties = propertynames(particles)
        print(io, "├── coriolis: $(typeof(model.coriolis))\n")
        print(io, "└── particles: $(length(particles)) Lagrangian particles with $(length(properties)) properties: $properties")
    end
end
