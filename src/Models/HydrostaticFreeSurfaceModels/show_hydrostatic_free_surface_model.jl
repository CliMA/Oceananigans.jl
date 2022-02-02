using Oceananigans.Utils: prettytime, ordered_dict_show

"""Show the innards of a `Model` in the REPL."""
function Base.show(io::IO, model::HydrostaticFreeSurfaceModel{TS, C, A}) where {TS, C, A}
    print(io, "HydrostaticFreeSurfaceModel{$A, $(eltype(model.grid))}",
        "(time = $(prettytime(model.clock.time)), iteration = $(model.clock.iteration)) \n",
        "├── grid: $(summary(model.grid))\n",
        "├── tracers: $(tracernames(model.tracers))\n",
        "├── closure: ", summary(model.closure), '\n',
        "├── buoyancy: ", summary(model.buoyancy), '\n')

    if model.free_surface !== nothing
        print(io, "├── free surface: ", typeof(model.free_surface).name.wrapper, " with gravitational acceleration $(model.free_surface.gravitational_acceleration) m s⁻²", '\n')

        if typeof(model.free_surface).name.wrapper == ImplicitFreeSurface
            print(io, "│   └── solver: ", string(typeof(model.free_surface.implicit_step_solver).name.name), '\n')
        end

        if typeof(model.free_surface).name.wrapper == SplitExplicitFreeSurface
            print(io, "│   └── number of substeps: $(model.free_surface.settings.substeps)", '\n')
        end
    end

    if isnothing(model.particles)
        print(io, "└── coriolis: $(typeof(model.coriolis))")
    else
        particles = model.particles.properties
        properties = propertynames(particles)
        print(io, "├── coriolis: $(typeof(model.coriolis))\n")
        print(io, "└── particles: $(length(particles)) Lagrangian particles with $(length(properties)) properties: $properties")
    end
end
