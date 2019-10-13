push!(LOAD_PATH,"../src/")
push!(LOAD_PATH,"../src/operators/")
push!(LOAD_PATH,"../src/turbulence_closures/")

using
    Documenter,
    Oceananigans,
    Oceananigans.Operators,
    Oceananigans.TurbulenceClosures

makedocs(
   modules = [Oceananigans, Oceananigans.Operators, Oceananigans.TurbulenceClosures],
   doctest = true,
   clean   = true,
 checkdocs = :all,
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                             canonical = "https://climate-machine.github.io/Oceananigans.jl/latest/"),
   authors = "Ali Ramadhan, Greg Wagner, John Marshall, Jean-Michel Campin, Chris Hill",
  sitename = "Oceananigans.jl",
     pages = ["Home"         => "index.md",
              "Examples"     => "examples.md",
              "Manual"       => ["equations.md",
                                 "algorithm.md"],
              "Verification" => "verification.md",
              "Gallery"      => "gallery.md",
              "Benchmarks"   => "benchmarks.md",
              "Internals"    => ["internal/grids.md",
                                 "internal/fields.md",
                                 "internal/operators.md"],
              "Index"        => "subject_index.md"]
)

deploydocs(repo = "github.com/climate-machine/Oceananigans.jl.git")
