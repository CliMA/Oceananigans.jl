push!(LOAD_PATH,"../src/")
push!(LOAD_PATH,"../src/operators/")
push!(LOAD_PATH,"../src/turbulence_closures/")

using
    Documenter,
    Literate,
    Oceananigans,
    Oceananigans.Operators,
    Oceananigans.TurbulenceClosures

####
#### Generate examples
####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = [
    "simple_diffusion.jl",
    "two_dimensional_turbulence.jl",
    # "ocean_wind_mixing_and_convection.jl",
    # "ocean_convection_with_plankton.jl",
    "internal_wave.jl"
]

for example in examples
    example_filepath = joinpath(EXAMPLES_DIR, example)
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter=true)
end

####
#### Build docs
####

makedocs(
   modules = [Oceananigans, Oceananigans.Operators, Oceananigans.TurbulenceClosures],
   doctest = true,
   clean   = true,
 checkdocs = :all,
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                             canonical = "https://climate-machine.github.io/Oceananigans.jl/latest/"),
   authors = "Ali Ramadhan, Greg Wagner, John Marshall, Jean-Michel Campin, Chris Hill",
  sitename = "Oceananigans.jl",
     pages = [
         "Home"   => "index.md",
         "Manual" => [
             "Continuous equations" => "manual/equations.md",
             "Numerical methods" => [
                 "Overview"               => "manual/overview.md",
                 "Finite volume method"   => "manual/finite_volume.md",
                 "Staggered grid"         => "manual/staggered_grid.md",
                 "Fractional step method" => "manual/fractional_step.md",
                 "Time stepping"          => "manual/time_stepping.md",
                 "Spatial operators"      => "manual/spatial_operators.md",
                 "Poisson solvers"        => "manual/poisson_solvers.md",
                 "Boundary conditions"    => "manual/boundary_conditions.md",
                 "Turbulence closures"    => "manual/turbulence_closures.md",
                 "Large eddy simulation"  => "manual/large_eddy_simulation.md"
             ],
             "Model setup" => "manual/model_setup.md",
             "Examples" => [
                 "One-dimensional diffusion"        => "generated/simple_diffusion.md",
                 "Two-dimensional turbulence"       => "generated/two_dimensional_turbulence.md",
                 # "Ocean wind mixing and convection" => "generated/ocean_wind_mixing_and_convection.md",
                 # "Ocean convection with plankton"   => "generated/ocean_convection_with_plankton.md",
                 "Internal wave"                    => "generated/internal_wave.md"
             ],
             "Verification" => [
                 "Taylor-Green vortex"     => "verification/taylor_green_vortex.md",
                 "Lid-driven cavity"       => "verification/lid_driven_cavity.md",
                 "Free convection"         => "verification/free_convection.md",
                 "Stratified Couette flow" => "verification/stratified_couette_flow.md"
             ]
         ],
         "Gallery"    => "gallery.md",
         "Benchmarks" => "benchmarks.md",
         "Library"    => "library.md",
         "Index"      => "subject_index.md"
     ]
)

deploydocs(repo = "github.com/climate-machine/Oceananigans.jl.git")
