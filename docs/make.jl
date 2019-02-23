using Documenter
using Oceananigans, Oceananigans.Operators

push!(LOAD_PATH,"../src/")

makedocs(
   modules = [Oceananigans],
   doctest = true,
   clean   = true,
 checkdocs = :all,
    format = :html,
   authors = "Ali Ramadhan",
  sitename = "Oceananigans.jl",
     pages = Any["Home" => "index.md"]
)

deploydocs(
    # deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo   = "github.com/ali-ramadhan/Oceananigans.jl.git",
    julia  = "1.1",
    osname = "linux"
)
