pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add Oceananigans to environment stack

using Documenter
using DocumenterCitations
using Literate
using Plots # to avoid capturing precompilation output by Literate
using Glob

using Oceananigans
using Oceananigans.Operators
using Oceananigans.Grids
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.TurbulenceClosures
using Oceananigans.TimeSteppers
using Oceananigans.AbstractOperations

using Oceananigans.BoundaryConditions: Flux, Value, Gradient, Open

bib_filepath = joinpath(dirname(@__FILE__), "oceananigans.bib")
bib = CitationBibliography(bib_filepath)

# Gotta set this environment variable when using the GR run-time on a remote machine.
# This happens as examples will use Plots.jl to make plots and movies.
# See: https://github.com/jheinen/GR.jl/issues/278

ENV["GKSwstype"] = "100"

#####
##### Generate examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = [
    "one_dimensional_diffusion.jl",
    "geostrophic_adjustment.jl",
    "two_dimensional_turbulence.jl",
    "internal_wave.jl",
    "convecting_plankton.jl",
    "ocean_wind_mixing_and_convection.jl",
    "langmuir_turbulence.jl",
    "eady_turbulence.jl",
    "kelvin_helmholtz_instability.jl",
    "shallow_water_Bickley_jet.jl"
]

for example in examples
    example_filepath = joinpath(EXAMPLES_DIR, example)
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter=true)
end

#####
##### Organize page hierarchies
#####

example_pages = [
    "One-dimensional diffusion"          => "generated/one_dimensional_diffusion.md",
    "Geostrophic adjustment"             => "generated/geostrophic_adjustment.md",
    "Two-dimensional turbulence"         => "generated/two_dimensional_turbulence.md",
    "Internal wave"                      => "generated/internal_wave.md",
    "Convecting plankton"                => "generated/convecting_plankton.md",
    "Ocean wind mixing and convection"   => "generated/ocean_wind_mixing_and_convection.md",
    "Langmuir turbulence"                => "generated/langmuir_turbulence.md",
    "Eady turbulence"                    => "generated/eady_turbulence.md",
    "Kelvin-Helmholtz instability"       => "generated/kelvin_helmholtz_instability.md",
    "Shallow water Bickley jet"          => "generated/shallow_water_Bickley_jet.md"
 ]

model_setup_pages = [
    "Overview" => "model_setup/overview.md",
    "Architecture" => "model_setup/architecture.md",
    "Number type" => "model_setup/number_type.md",
    "Grid" => "model_setup/grids.md",
    "Clock" => "model_setup/clock.md",
    "Coriolis (rotation)" => "model_setup/coriolis.md",
    "Tracers" => "model_setup/tracers.md",
    "Buoyancy models and equation of state" => "model_setup/buoyancy_and_equation_of_state.md",
    "Boundary conditions" => "model_setup/boundary_conditions.md",
    "Forcing functions" => "model_setup/forcing_functions.md",
    "Background fields" => "model_setup/background_fields.md",
    "Turbulent diffusivity closures and LES models" => "model_setup/turbulent_diffusivity_closures_and_les_models.md",
    "Lagrangian particles" => "model_setup/lagrangian_particles.md",
    "Diagnostics" => "model_setup/diagnostics.md",
    "Output writers" => "model_setup/output_writers.md",
    "Checkpointing" => "model_setup/checkpointing.md",
    "Setting initial conditions" => "model_setup/setting_initial_conditions.md"
]

physics_pages = [
    "Coordinate system and notation" => "physics/notation.md",
    "Boussinesq approximation" => "physics/boussinesq.md",
    "`IncompressibleModel`" => [
        "Incompressible model" => "physics/incompressible_model.md",
        ],
    "`HydrostaticFreeSurfaceModel`" => [
        "Hydrostatic model with a free surface" => "physics/hydrostatic_free_surface_model.md"
        ],
    "`ShallowWaterModel`" => [
        "Shallow water model" => "physics/shallow_water_model.md"
        ],
    "Buoyancy models and equations of state" => "physics/buoyancy_and_equations_of_state.md",
    "Coriolis forces" => "physics/coriolis_forces.md",
    "Turbulence closures" => "physics/turbulence_closures.md",
    "Surface gravity waves and the Craik-Leibovich approximation" => "physics/surface_gravity_waves.md"
]

numerical_pages = [
    "Pressure decomposition" => "numerical_implementation/pressure_decomposition.md",
    "Time stepping" => "numerical_implementation/time_stepping.md",
    "Finite volume method" => "numerical_implementation/finite_volume.md",
    "Spatial operators" => "numerical_implementation/spatial_operators.md",
    "Boundary conditions" => "numerical_implementation/boundary_conditions.md",
    "Poisson solvers" => "numerical_implementation/poisson_solvers.md",
    "Large eddy simulation" => "numerical_implementation/large_eddy_simulation.md"
]

appendix_pages = [
    "Staggered grid" => "appendix/staggered_grid.md",
    "Fractional step method" => "appendix/fractional_step.md",
    "Convergence tests" => "appendix/convergence_tests.md",
    "Performance benchmarks" => "appendix/benchmarks.md",
    "Library" => "appendix/library.md",
    "Function index" => "appendix/function_index.md",
]

pages = [
    "Home" => "index.md",
    "Installation instructions" => "installation_instructions.md",
    "Using GPUs" => "using_gpus.md",
    "Examples" => example_pages,
    "Physics" => physics_pages,
    "Numerical implementation" => numerical_pages,
    "Model setup" => model_setup_pages,
    "Simulation tips" => "simulation_tips.md",
    "Contributor's guide" => "contributing.md",
    "Gallery" => "gallery.md",
    "References" => "references.md",
    "Appendix" => appendix_pages,
]

#####
##### Build and deploy docs
#####

format = Documenter.HTML(
    collapselevel = 1,
       prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://clima.github.io/OceananigansDocumentation/stable/",
       mathengine = MathJax3()
)

makedocs(bib,
  sitename = "Oceananigans.jl",
   authors = "Ali Ramadhan, Gregory Wagner, John Marshall, Jean-Michel Campin, Chris Hill",
    format = format,
     pages = pages,
   modules = [Oceananigans],
   doctest = true,
    strict = true,
     clean = true,
 checkdocs = :none # Should fix our docstring so we can use checkdocs=:exports with strict=true.
)

deploydocs(
          repo = "github.com/CliMA/OceananigansDocumentation.git",
      versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"],
  push_preview = true
)

@info "Cleaning up temporary .jld2 and .nc files created by doctests..."
for file in vcat(glob("docs/*.jld2"), glob("docs/*.nc"))
    rm(file)
end
