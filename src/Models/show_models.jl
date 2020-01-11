using Oceananigans.Grids: short_show
using Oceananigans.Utils: prettytime, ordered_dict_show

"""Show the innards of a `Model` in the REPL."""
Base.show(io::IO, model::Model) =
    print(io, "Oceananigans.Model on a ", typeof(model.architecture), " architecture ",
                                          "(time = ",  prettytime(model.clock.time),
                                          ", iteration = ", model.clock.iteration, ") \n",
              "├── grid: ", short_show(model.grid), '\n',
              "├── tracers: ", tracernames(model.tracers), '\n',
              "├── closure: ", typeof(model.closure), '\n',
              "├── buoyancy: ", typeof(model.buoyancy), '\n',
              "├── coriolis: ", typeof(model.coriolis), '\n',
              "├── output writers: ", ordered_dict_show(model.output_writers, "│"), '\n',
              "└── diagnostics: ", ordered_dict_show(model.diagnostics, " "))
