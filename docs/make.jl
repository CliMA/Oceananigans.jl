using Documenter
using Oceananigans, Oceananigans.Operators

push!(LOAD_PATH,"../src/")

makedocs(sitename="Oceananigans.jl")

deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo   = "github.com/ali-ramadhan/Oceananigans.jl.git",
    julia  = "1.0",
    osname = "linux"
)
