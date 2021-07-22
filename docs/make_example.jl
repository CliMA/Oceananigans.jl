#=
This script can be used to build the Documentation only with a few examples (e.g., an example 
a developer is currently working on). This makes previewing how the example will look like 
in the actual documentation much faster. To use the script, modify it to include the example
you are working on and then run:

$ julia --project=docs/ -e 'using Pkg; Pkg.instantiate(); Pkg.develop(PackageSpec(path=pwd()))'; julia --project=docs/ docs/make_example.jl

from the repo's home directory and then open `docs/build/index.html` with your favorite browser.
=#

push!(LOAD_PATH, "..")

using Documenter
using Literate
using Plots  # to avoid capturing precompilation output by Literate

using Oceananigans
using Oceananigans.Operators
using Oceananigans.Grids
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.TurbulenceClosures
using Oceananigans.TimeSteppers
using Oceananigans.AbstractOperations

#####
##### Generate examples
#####

# Gotta set this environment variable when using the GR run-time on CI machines.
# This is needed as examples use Plots.jl to make plots and movies.
# See: https://github.com/jheinen/GR.jl/issues/278

ENV["GKSwstype"] = "100"

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = [
            "horizontal_convection.jl"
            ]

for example in examples
    example_filepath = joinpath(EXAMPLES_DIR, example)
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter=true)
end

#####
##### Organize page hierarchies
#####

example_pages = [
                 "Horizontal convection" => "generated/horizontal_convection.md"
                 ]

pages = [
         "Home" => "index.md",
         "Examples" => example_pages
        ]

#####
##### Build and deploy docs
#####

format = Documenter.HTML(collapselevel = 1,
                            prettyurls = false
                        )

makedocs(sitename = "Oceananigans.jl",
          authors = "Ali Ramadhan, Gregory Wagner, John Marshall, Jean-Michel Campin, Chris Hill, Navid Constantinou",
           format = format,
            pages = pages,
          modules = [Oceananigans],
          doctest = false,
           strict = false,
            clean = true,
        checkdocs = :none  # Should fix our docstring so we can use checkdocs=:exports with strict=true.
        )
