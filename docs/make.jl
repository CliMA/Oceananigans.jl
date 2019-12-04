push!(LOAD_PATH, "..")

using
    Documenter,
    Literate,
    Oceananigans,
    Oceananigans.Operators,
    Oceananigans.TurbulenceClosures

#####
#####Generate examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = [
    "simple_diffusion.jl",
    "two_dimensional_turbulence.jl",
    "ocean_wind_mixing_and_convection.jl",
    "ocean_convection_with_plankton.jl",
    "internal_wave.jl"
]

for example in examples
    example_filepath = joinpath(EXAMPLES_DIR, example)
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter=true)
end

#####
#####Build docs
#####

makedocs(
   modules = [Oceananigans, Oceananigans.Operators, Oceananigans.TurbulenceClosures],
   doctest = true,
   clean   = true,
 checkdocs = :all,
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                             canonical = "https://climate-machine.github.io/Oceananigans.jl/latest/"),
   authors = "Ali Ramadhan, Gregory Wagner, John Marshall, Jean-Michel Campin, Chris Hill",
  sitename = "Oceananigans.jl",
     pages = [
         "Home"   => "index.md",
         "Manual" => [
             #"Model setup" => "manual/model_setup.md",
             "Examples" => [
                 "One-dimensional diffusion"        => "generated/simple_diffusion.md",
                 "Two-dimensional turbulence"       => "generated/two_dimensional_turbulence.md",
                 "Ocean wind mixing and convection" => "generated/ocean_wind_mixing_and_convection.md",
                 "Ocean convection with plankton"   => "generated/ocean_convection_with_plankton.md",
                 "Internal wave"                    => "generated/internal_wave.md"
             ],
             "Physics" => "manual/physics.md",
             "Numerical implementation" => [
                 #"Overview"               => "manual/overview.md",
                 "Pressure decomposition" => "manual/pressure_decomposition.md",
                 "Time stepping"          => "manual/time_stepping.md",
                 "Finite volume method"   => "manual/finite_volume.md",
                 "Spatial operators"      => "manual/spatial_operators.md",
                 "Boundary conditions"    => "manual/boundary_conditions.md",
                 "Poisson solvers"        => "manual/poisson_solvers.md",
                 "Large eddy simulation"  => "manual/large_eddy_simulation.md"
             ],
             "Verification" => [
                 "Taylor-Green vortex"     => "verification/taylor_green_vortex.md",
                 "Stratified Couette flow" => "verification/stratified_couette_flow.md"
             ],
             #"Appendix" => [
                 #"Staggered grid"         => "manual/staggered_grid.md",
                 #"Fractional step method" => "manual/fractional_step.md",
             #],
         ],
         "Gallery"    => "gallery.md",
         "Benchmarks" => "benchmarks.md",
         "Library"    => "library.md",
         "Index"      => "subject_index.md"
     ]
)

deploydocs(repo = "github.com/climate-machine/Oceananigans.jl.git")

#####
#####Delete leftover JLD2 files.
#####See: https://github.com/climate-machine/Oceananigans.jl/issues/509
#####

const GENERATED_DIR = joinpath(@__DIR__, "build/generated")

@info "Deleting leftover JLD2 files..."
leftovers = filter(x -> occursin(".jld2", x), readdir(GENERATED_DIR))
for fname in leftovers
    fpath = joinpath(GENERATED_DIR, fname)
    rm(fpath, force=true)
    @info "Deleted: $fpath"
end

