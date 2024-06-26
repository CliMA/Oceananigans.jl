using Distributed
Distributed.addprocs(2)

@everywhere begin
    using Documenter
    using DocumenterCitations
    using Literate
    using Printf

    using CairoMakie # to avoid capturing precompilation output by Literate
    CairoMakie.activate!(type = "svg")

    using Oceananigans
    using Oceananigans.Operators
    using Oceananigans.Diagnostics
    using Oceananigans.OutputWriters
    using Oceananigans.TurbulenceClosures
    using Oceananigans.TimeSteppers
    using Oceananigans.AbstractOperations

    using Oceananigans.BoundaryConditions: Flux, Value, Gradient, Open

    bib_filepath = joinpath(dirname(@__FILE__), "oceananigans.bib")
    bib = CitationBibliography(bib_filepath, style=:authoryear)

    #####
    ##### Generate examples
    #####

    const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
    const OUTPUT_DIR   = joinpath(@__DIR__, "src/literated")

    # The examples that take longer to run should be first. This ensures thats
    # docs built using extra workers is as efficient as possible.
    example_scripts = [
        "internal_tide.jl",
        "shallow_water_Bickley_jet.jl",
        "kelvin_helmholtz_instability.jl",
        "horizontal_convection.jl",
        "langmuir_turbulence.jl",
        "baroclinic_adjustment.jl",
        "tilted_bottom_boundary_layer.jl",
        "convecting_plankton.jl",
        "ocean_wind_mixing_and_convection.jl",
        "two_dimensional_turbulence.jl",
        "one_dimensional_diffusion.jl",
        "internal_wave.jl",
    ]

    @info string("Executing the examples using ", Distributed.nprocs(), " processes")
end

Distributed.pmap(1:length(example_scripts)) do n
    example = example_scripts[n]
    example_filepath = joinpath(EXAMPLES_DIR, example)
    withenv("JULIA_DEBUG" => "Literate") do
        start_time = time_ns()
        Literate.markdown(example_filepath, OUTPUT_DIR;
                          flavor = Literate.DocumenterFlavor(), execute = true)
        elapsed = 1e-9 * (time_ns() - start_time)
        @info @sprintf("%s example took %s to build.", example, prettytime(elapsed))
    end
end

Distributed.rmprocs()

#####
##### Organize page hierarchies
#####

example_pages = [
    "One-dimensional diffusion"        => "literated/one_dimensional_diffusion.md",
    "Two-dimensional turbulence"       => "literated/two_dimensional_turbulence.md",
    "Internal wave"                    => "literated/internal_wave.md",
    "Internal tide by a seamount"      => "literated/internal_tide.md",
    "Convecting plankton"              => "literated/convecting_plankton.md",
    "Ocean wind mixing and convection" => "literated/ocean_wind_mixing_and_convection.md",
    "Langmuir turbulence"              => "literated/langmuir_turbulence.md",
    "Baroclinic adjustment"            => "literated/baroclinic_adjustment.md",
    "Kelvin-Helmholtz instability"     => "literated/kelvin_helmholtz_instability.md",
    "Shallow water Bickley jet"        => "literated/shallow_water_Bickley_jet.md",
    "Horizontal convection"            => "literated/horizontal_convection.md",
    "Tilted bottom boundary layer"     => "literated/tilted_bottom_boundary_layer.md"
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
    "Callbacks" => "model_setup/callbacks.md",
    "Output writers" => "model_setup/output_writers.md",
    "Checkpointing" => "model_setup/checkpointing.md",
    "Setting initial conditions" => "model_setup/setting_initial_conditions.md"
]

physics_pages = [
    "Coordinate system and notation" => "physics/notation.md",
    "Boussinesq approximation" => "physics/boussinesq.md",
    "`NonhydrostaticModel`" => [
        "Nonhydrostatic model" => "physics/nonhydrostatic_model.md",
        ],
    "`HydrostaticFreeSurfaceModel`" => [
        "Hydrostatic model with a free surface" => "physics/hydrostatic_free_surface_model.md"
        ],
    "`ShallowWaterModel`" => [
        "Shallow water model" => "physics/shallow_water_model.md"
        ],
    "Boundary conditions" => "physics/boundary_conditions.md",
    "Buoyancy models and equations of state" => "physics/buoyancy_and_equations_of_state.md",
    "Coriolis forces" => "physics/coriolis_forces.md",
    "Turbulence closures" => "physics/turbulence_closures.md",
    "Surface gravity waves and the Craik-Leibovich approximation" => "physics/surface_gravity_waves.md"
]

numerical_pages = [
    "Finite volume method" => "numerical_implementation/finite_volume.md",
    "Spatial operators" => "numerical_implementation/spatial_operators.md",
    "Pressure decomposition" => "numerical_implementation/pressure_decomposition.md",
    "Time stepping" => "numerical_implementation/time_stepping.md",
    "Boundary conditions" => "numerical_implementation/boundary_conditions.md",
    "Elliptic solvers" => "numerical_implementation/elliptic_solvers.md",
    "Large eddy simulation" => "numerical_implementation/large_eddy_simulation.md"
]

appendix_pages = [
    "Staggered grid" => "appendix/staggered_grid.md",
    "Fractional step method" => "appendix/fractional_step.md",
    "Convergence tests" => "appendix/convergence_tests.md",
    "Performance benchmarks" => "appendix/benchmarks.md",
    "Library" => "appendix/library.md",
    "Function index" => "appendix/function_index.md"
]

pages = [
    "Home" => "index.md",
    "Quick start" => "quick_start.md",
    "Examples" => example_pages,
    "Physics" => physics_pages,
    "Numerical implementation" => numerical_pages,
    "Model setup" => model_setup_pages,
    "Simulation tips" => "simulation_tips.md",
    "Contributor's guide" => "contributing.md",
    "Gallery" => "gallery.md",
    "References" => "references.md",
    "Appendix" => appendix_pages
]

#####
##### Build and deploy docs
#####

format = Documenter.HTML(collapselevel = 1,
                         prettyurls = get(ENV, "CI", nothing) == "true",
                         canonical = "https://clima.github.io/OceananigansDocumentation/stable/",
                         mathengine = MathJax3(),
                         size_threshold = 2^20,
                         assets = String["assets/citations.css"])

DocMeta.setdocmeta!(Oceananigans, :DocTestSetup, :(using Oceananigans); recursive=true)

makedocs(sitename = "Oceananigans.jl",
         authors = "Climate Modeling Alliance and contributors",
         format = format,
         pages = pages,
         plugins = [bib],
         modules = [Oceananigans],
         warnonly = [:cross_references],
         doctest = true, # set to false to speed things up
         clean = true,
         checkdocs = :exports) # set to :none to speed things up

@info "Clean up temporary .jld2 and .nc output created by doctests or literated examples..."

"""
    recursive_find(directory, pattern)

Return list of filepaths within `directory` that contains the `pattern::Regex`.
"""
recursive_find(directory, pattern) =
    mapreduce(vcat, walkdir(directory)) do (root, dirs, files)
        joinpath.(root, filter(contains(pattern), files))
    end

files = []
for pattern in [r"\.jld2", r"\.nc"]
    global files = vcat(files, recursive_find(@__DIR__, pattern))
end

for file in files
    rm(file)
end

deploydocs(repo = "github.com/CliMA/OceananigansDocumentation.git",
           versions = ["stable" => "v^", "dev" => "dev", "v#.#.#"],
           forcepush = true,
           push_preview = false,
           devbranch = "main")
