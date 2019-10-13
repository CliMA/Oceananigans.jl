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
     pages = ["Home"       => "index.md",
              "Manual"     => ["Continuous equations" => "manual/equations.md",
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
                                              "Rising thermal bubble" => "examples/rising_thermal_bubble.md"
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
