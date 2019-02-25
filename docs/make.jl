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
    format = Documenter.HTML(prettyurls=true,
                             canonical="https://ali-ramadhan.github.io/Oceananigans.jl/latest"),
   authors = "Ali Ramadhan, John Marshall, Jean-Michel Campin, Chris Hill",
  sitename = "Oceananigans.jl",
     pages = ["Home" => "index.md",
              "Examples" => "examples.md",
              "Numerical algorithm" => "algorithm.md",
              "Internals" => ["internal/grids.md",
                              "internal/fields.md",
                              "internal/operators.md"],
              "Index" => "subject_index.md"]
)

deploydocs(repo = "github.com/ali-ramadhan/Oceananigans.jl.git")
