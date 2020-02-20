push!(LOAD_PATH, "..")

using
    Documenter,
    Literate,
    Oceananigans,
    Oceananigans.Operators,
    Oceananigans.Grids,
    Oceananigans.Diagnostics,
    Oceananigans.OutputWriters,
    Oceananigans.TurbulenceClosures,
    Oceananigans.TimeSteppers,
    Oceananigans.AbstractOperations

#####
##### Generate examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = [
    "one_dimensional_diffusion.jl",
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

# Set up a timer to print a dot '.' every 60 seconds. This is to avoid Travis CI
# timing out when building demanding Literate.jl examples.
Timer(t -> println("."), 0, interval=60)

format = Documenter.HTML(
    collapselevel = 1,
       prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://climate-machine.github.io/Oceananigans.jl/latest/"
)

makedocs(
   modules = [Oceananigans,
              Oceananigans.Grids,
              Oceananigans.Operators,
              Oceananigans.Diagnostics,
              Oceananigans.OutputWriters,
              Oceananigans.TimeSteppers,
              Oceananigans.TurbulenceClosures,
              Oceananigans.AbstractOperations],
   doctest = true,
   clean   = true,
 checkdocs = :all,
    format = format,
   authors = "Ali Ramadhan, Gregory Wagner, John Marshall, Jean-Michel Campin, Chris Hill",
  sitename = "Oceananigans.jl",
     pages = [
                              "Home" => "index.md",
         "Installation instructions" => "installation_instructions.md",
                        "Using GPUs" => "using_gpus.md",

         "Examples" => [
             "One-dimensional diffusion"        => "generated/one_dimensional_diffusion.md",
             "Two-dimensional turbulence"       => "generated/two_dimensional_turbulence.md",
             "Ocean wind mixing and convection" => "generated/ocean_wind_mixing_and_convection.md",
             "Ocean convection with plankton"   => "generated/ocean_convection_with_plankton.md",
             "Internal wave"                    => "generated/internal_wave.md"
         ],

         "Model setup" => [
                       "Overview" => "model_setup/overview.md",
                   "Architecture" => "model_setup/architecture.md",
                    "Number type" => "model_setup/number_type.md",
                           "Grid" => "model_setup/grids.md",
                          "Clock" => "model_setup/clock.md",
            "Coriolis (rotation)" => "model_setup/coriolis.md",
                        "Tracers" => "model_setup/tracers.md",
            "Buoyancy and equation of state" =>
                                     "model_setup/buoyancy_and_equation_of_state.md",
            "Boundary conditions" => "model_setup/boundary_conditions.md",
              "Forcing functions" => "model_setup/forcing_functions.md",
               "Model parameters" => "model_setup/model_parameters.md",
            "Turbulent diffusivity closures and LES models" =>
                                     "model_setup/turbulent_diffusivity_closures_and_les_models.md",
                    "Diagnostics" => "model_setup/diagnostics.md",
                 "Output writers" => "model_setup/output_writers.md",
                  "Checkpointing" => "model_setup/checkpointing.md",
                  "Time stepping" => "model_setup/time_stepping.md",
            "Setting initial conditions" =>
                                     "model_setup/setting_initial_conditions.md"
         ],

         "Physics" => [
            "Navier-Stokes and tracer conservation equations" =>
                                                "physics/navier_stokes_and_tracer_conservation.md",
            "Coordinate system and notation" => "physics/coordinate_system_and_notation.md",
              "The Boussinesq approximation" => "physics/boussinesq_approximation.md",
                           "Coriolis forces" => "physics/coriolis_forces.md",
            "Buoyancy model and equations of state" =>
                                                "physics/buoyancy_and_equations_of_state.md",
                       "Turbulence closures" => "physics/turbulence_closures.md",
            "Surface gravity waves and the Craik-Leibovich approximation" =>
                                                "physics/surface_gravity_waves.md"
        ],

         "Numerical implementation" => [
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

                        "Gallery" => "gallery.md",
         "Performance benchmarks" => "benchmarks.md",
            "Contributor's guide" => "contributing.md",
                        "Library" => "library.md",

         "Appendix" => [
             "Staggered grid"         => "appendix/staggered_grid.md",
             "Fractional step method" => "appendix/fractional_step.md",
         ],

         "Function index" => "function_index.md"
     ]
)

deploydocs(repo = "github.com/climate-machine/Oceananigans.jl.git")
