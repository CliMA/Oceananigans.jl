push!(LOAD_PATH,"../src/")  # Add package to PATH.

using Documenter, DocumenterMarkdown
using Oceananigans, Oceananigans.Operators

makedocs(
   modules = [Oceananigans, Oceananigans.Operators],
   doctest = true,
   clean   = true,
 checkdocs = :all,
    format = DocumenterMarkdown.Markdown(),
   authors = "Ali Ramadhan",
  sitename = "Oceananigans.jl",
     pages = Any["Home" => "index.md"]
)

deploydocs(
    repo   = "github.com/ali-ramadhan/Oceananigans.jl.git",
    deps   = Deps.pip("mkdocs==0.16.3", "mkdocs-material", "pygments", "python-markdown-math"),
    make   = () -> run(`mkdocs build`),
    target = "site"
)
