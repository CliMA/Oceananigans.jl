using Documenter
using Oceananigans, Oceananigans.Operators

push!(LOAD_PATH,"../src/")

makedocs(sitename="Oceananigans.jl")

deploydocs(repo = "github.com/ali-ramadhan/Oceananigans.jl.git",)
