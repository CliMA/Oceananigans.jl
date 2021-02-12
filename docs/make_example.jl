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

# Gotta set this environment variable when using the GR run-time on Travis CI.
# This happens as examples will use Plots.jl to make plots and movies.
# See: https://github.com/jheinen/GR.jl/issues/278
ENV["GKSwstype"] = "100"

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = [
           # "internal_wave.jl",
            "geostrophic_adjustment.jl"
           ]

for example in examples
    example_filepath = joinpath(EXAMPLES_DIR, example)
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter=true)
end

#####
##### Organize page hierarchies
#####

example_pages = [
                 #"Internal wave"                    => "generated/internal_wave.md",
                 "Geostrophic adjustment"            => "generated/geostrophic_adjustment.md"
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
          authors = "Ali Ramadhan, Gregory Wagner, John Marshall, Jean-Michel Campin, Chris Hill",
           format = format,
            pages = pages,
          modules = [Oceananigans],
          doctest = false,
           strict = false,
            clean = true,
        checkdocs = :none  # Should fix our docstring so we can use checkdocs=:exports with strict=true.
        )
