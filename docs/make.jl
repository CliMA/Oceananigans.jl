push!(LOAD_PATH, "..")

using
    Documenter,
    Literate,
    Oceananigans,
    Oceananigans.Operators,
    Oceananigans.TurbulenceClosures

#####
##### Generate examples
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
##### Build and deploy docs
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
         "Examples" => [
             "One-dimensional diffusion"        => "generated/simple_diffusion.md",
             "Two-dimensional turbulence"       => "generated/two_dimensional_turbulence.md",
             "Ocean wind mixing and convection" => "generated/ocean_wind_mixing_and_convection.md",
             "Ocean convection with plankton"   => "generated/ocean_convection_with_plankton.md",
             "Internal wave"                    => "generated/internal_wave.md"
         ],
         "Model setup" => "model_setup.md",
         "Physics" => "physics.md",
         "Numerical implementation" => [
             # "Overview"               => "manual/overview.md",
             "Pressure decomposition" => "numerical_implementation/pressure_decomposition.md",
             "Time stepping"          => "numerical_implementation/time_stepping.md",
             "Finite volume method"   => "numerical_implementation/finite_volume.md",
             "Spatial operators"      => "numerical_implementation/spatial_operators.md",
             "Boundary conditions"    => "numerical_implementation/boundary_conditions.md",
             "Poisson solvers"        => "numerical_implementation/poisson_solvers.md",
             "Large eddy simulation"  => "numerical_implementation/large_eddy_simulation.md"
         ],
         "Verification experiments" => [
             "Taylor-Green vortex"     => "verification/taylor_green_vortex.md",
             "Stratified Couette flow" => "verification/stratified_couette_flow.md"
         ],
         "Gallery"    => "gallery.md",
         "Performance benchmarks" => "benchmarks.md",
         "Library"    => "library.md",
         "Appendix" => [
             "Staggered grid"         => "appendix/staggered_grid.md",
             "Fractional step method" => "appendix/fractional_step.md",
         ],
         "Index"      => "subject_index.md"
     ]
)

deploydocs(repo = "github.com/climate-machine/Oceananigans.jl.git")
