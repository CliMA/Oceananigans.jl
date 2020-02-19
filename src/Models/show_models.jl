using Oceananigans.Grids: short_show
using Oceananigans.Utils: prettytime, ordered_dict_show

"""Show the innards of a `Model` in the REPL."""
Base.show(io::IO, model::IncompressibleModel) =
    print(io, "Oceananigans.IncompressibleModel on a $(typeof(model.architecture)) architecture ",
        "(time = $(prettytime(model.clock.time)), iteration = $(model.clock.iteration)) \n",
        "├── grid: $(short_show(model.grid))\n",
        "├── tracers: $(tracernames(model.tracers))\n",
        "├── closure: $(typeof(model.closure))\n",
        "├── buoyancy: $(typeof(model.buoyancy))\n",
        "└── coriolis: $(typeof(model.coriolis))")
