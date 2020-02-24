using Oceananigans.Grids: short_show
using Oceananigans.Utils: prettytime, ordered_dict_show

"""Show the innards of a `Model` in the REPL."""
Base.show(io::IO, model::IncompressibleModel{TS, C, A}) where {TS, C, A} =
    print(io, "IncompressibleModel{$A, $(eltype(model.grid))} ",
        "(time = $(prettytime(model.clock.time)), iteration = $(model.clock.iteration)) \n",
        "├── grid: $(short_show(model.grid))\n",
        "├── tracers: $(tracernames(model.tracers))\n",
        "├── closure: $(typeof(model.closure))\n",
        "├── buoyancy: $(typeof(model.buoyancy))\n",
        "└── coriolis: $(typeof(model.coriolis))")
