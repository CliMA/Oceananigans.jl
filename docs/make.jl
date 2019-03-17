push!(LOAD_PATH,"../src/")
push!(LOAD_PATH,"../src/operators/")

using Documenter
using Oceananigans, Oceananigans.Operators

makedocs(
   modules = [Oceananigans, Oceananigans.Operators],
   doctest = true,
   clean   = true,
 checkdocs = :all,
    assets = ["assets/invenia.css"],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                             canonical = "https://climate-machine.github.io/Oceananigans.jl/latest/"),
   authors = "Ali Ramadhan, John Marshall, Jean-Michel Campin, Chris Hill",
  sitename = "Oceananigans.jl",
     pages = ["Home" => "index.md",
              "Examples" => "examples.md",
              "Numerical algorithm" => "algorithm.md",
              "Performance benchmarks" => "benchmarks.md",
              "Internals" => ["internal/grids.md",
                              "internal/fields.md",
                              "internal/operators.md"],
              "Index" => "subject_index.md"]
)

deploydocs(repo = "github.com/climate-machine/Oceananigans.jl.git")
