using Oceananigans.Utils: prettytime
using Oceananigans: short_show

"""Show the innards of a `Model` in the REPL."""
Base.show(io::IO, model::ShallowWaterModel{TS, C, A}) where {TS, C, A} =
    print(io, "ShallowWaterModel{$A, $(eltype(model.grid))}",
        "(time = $(prettytime(model.clock.time)), iteration = $(model.clock.iteration)) \n",
        "├── grid: $(short_show(model.grid))\n",
        "├── tracers: $(tracernames(model.tracers))\n",
        "└── coriolis: $(typeof(model.coriolis))")
