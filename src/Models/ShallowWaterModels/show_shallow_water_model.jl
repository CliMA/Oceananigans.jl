using Oceananigans.Utils: prettytime

"""Show the innards of a `Model` in the REPL."""
Base.show(io::IO, model::ShallowWaterModel{G, A, T}) where {G, A, T} =
    print(io, "ShallowWaterModel{$(Base.typename(A)), $T}",
        "(time = $(prettytime(model.clock.time)), iteration = $(model.clock.iteration)) \n",
        "├── grid: $(summary(model.grid))\n",
        "├── tracers: $(tracernames(model.tracers))\n",
        "└── coriolis: $(typeof(model.coriolis))")
