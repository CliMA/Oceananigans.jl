using Oceananigans.Utils: prettytime

"""Show the innards of a `Model` in the REPL."""
function Base.show(io::IO, model::ShallowWaterModel{G, A, T}) where {G, A, T}
    TS = nameof(typeof(model.timestepper))

    print(io, "ShallowWaterModel{$(Base.typename(A)), $T}",
        "(time = $(prettytime(model.clock.time)), iteration = $(model.clock.iteration)) \n",
        "├── grid: $(summary(model.grid))\n",
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
        "├── tracers: $(tracernames(model.tracers))\n",
        "└── coriolis: $(typeof(model.coriolis))")
end
