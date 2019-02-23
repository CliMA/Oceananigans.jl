push!(LOAD_PATH,"../src/")  # Add package to PATH.

using Documenter
using Oceananigans, Oceananigans.Operators

makedocs(
   modules = [Oceananigans, Oceananigans.Operators],
   doctest = true,
   clean   = true,
 checkdocs = :all,
    assets = ["assets/invenia.css"],
    format = Documenter.HTML(prettyurls=true,
                             canonical="https://juliadocs.github.io/Documenter.jl/stable"),
   authors = "Ali Ramadhan",
  sitename = "Oceananigans.jl",
     pages = Any["Home" => "index.md"]
)

deploydocs(repo = "github.com/ali-ramadhan/Oceananigans.jl.git")
