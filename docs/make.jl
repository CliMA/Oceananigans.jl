using Distributed
Distributed.addprocs(2)

@everywhere begin
    using Documenter
    using DocumenterCitations
    using Literate
    using Printf
    using CUDA

    using CairoMakie # to avoid capturing precompilation output by Literate
    set_theme!(Theme(fontsize=20))
    CairoMakie.activate!(type = "png")

    using NCDatasets
    using XESMF

    using Oceananigans
    using Oceananigans.AbstractOperations
    using Oceananigans.Operators
    using Oceananigans.Diagnostics
    using Oceananigans.OutputWriters
    using Oceananigans.TimeSteppers
    using Oceananigans.TurbulenceClosures
    using Oceananigans.BoundaryConditions: Flux, Value, Gradient, Open

    bib_filepath = joinpath(dirname(@__FILE__), "oceananigans.bib")
    bib = CitationBibliography(bib_filepath, style=:authoryear)

    #####
    ##### Generate examples
    #####

    const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
    const OUTPUT_DIR   = joinpath(@__DIR__, "src/literated")

    # The examples that take longer to run should be first. This ensures that the
    # docs built which extra workers is as efficient as possible.
    example_scripts = [
        "spherical_baroclinic_instability.jl",
        "internal_tide.jl",
        "langmuir_turbulence.jl",
        "shallow_water_Bickley_jet.jl",
        "ocean_wind_mixing_and_convection.jl",
        "kelvin_helmholtz_instability.jl",
        "horizontal_convection.jl",
        "baroclinic_adjustment.jl",
        "tilted_bottom_boundary_layer.jl",
        "convecting_plankton.jl",
        "hydrostatic_lock_exchange.jl",
        "two_dimensional_turbulence.jl",
        "one_dimensional_diffusion.jl",
        "internal_wave.jl",
    ]
end

# We'll append the following postamble to the literate examples, to include
# information about the computing environment used to run them.
example_postamble = """

# ---

# ### Julia version and environment information
#
# This example was executed with the following version of Julia:

using InteractiveUtils: versioninfo
versioninfo()

# These were the top-level packages installed in the environment:

import Pkg
Pkg.status()
"""

@info string("Executing the examples using ", Distributed.nprocs(), " processes")

Distributed.pmap(1:length(example_scripts)) do n
    example = example_scripts[n]
    example_filepath = joinpath(EXAMPLES_DIR, example)
    withenv("JULIA_DEBUG" => "Literate") do
        start_time = time_ns()
        Literate.markdown(example_filepath, OUTPUT_DIR;
                          preprocess = content -> content * example_postamble,
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
    "One-dimensional diffusion"             => "literated/one_dimensional_diffusion.md",
    "Two-dimensional turbulence"            => "literated/two_dimensional_turbulence.md",
    "Internal wave"                         => "literated/internal_wave.md",
    "Internal tide by a seamount"           => "literated/internal_tide.md",
    "Convecting plankton"                   => "literated/convecting_plankton.md",
    "Ocean wind mixing and convection"      => "literated/ocean_wind_mixing_and_convection.md",
    "Langmuir turbulence"                   => "literated/langmuir_turbulence.md",
    "Baroclinic adjustment"                 => "literated/baroclinic_adjustment.md",
    "Kelvin-Helmholtz instability"          => "literated/kelvin_helmholtz_instability.md",
    "Hydrostatic lock exchange with CATKE"  => "literated/hydrostatic_lock_exchange.md",
    "Shallow water Bickley jet"             => "literated/shallow_water_Bickley_jet.md",
    "Horizontal convection"                 => "literated/horizontal_convection.md",
    "Tilted bottom boundary layer"          => "literated/tilted_bottom_boundary_layer.md",
    "Spherical baroclinic instability"      => "literated/spherical_baroclinic_instability.md"
]

model_pages = [
    "Overview" => "models/models_overview.md",
    "Coriolis forces" => "models/coriolis.md",
    "Buoyancy and equations of state" => "models/buoyancy_and_equation_of_state.md",
    "Stokes drift" => "models/stokes_drift.md",
    "Turbulence closures" => "models/turbulence_closures.md",
    "Boundary conditions" => "models/boundary_conditions.md",
    "Forcings" => "models/forcing_functions.md",
    "Lagrangian particles" => "models/lagrangian_particles.md",
    "Background fields" => "models/background_fields.md",
]

simulation_pages = [
    "Overview" => "simulations/simulations_overview.md",
    "Callbacks" => "simulations/callbacks.md",
    "Schedules" => "simulations/schedules.md",
    "Output writers" => "simulations/output_writers.md",
    "Checkpointing" => "simulations/checkpointing.md",
]

physics_pages = [
    "Coordinate systems" => "physics/coordinate_systems.md",
    "Boussinesq approximation" => "physics/boussinesq.md",
    "`NonhydrostaticModel`" => [
        "Nonhydrostatic model" => "physics/nonhydrostatic_model.md",
        ],
    "`HydrostaticFreeSurfaceModel`" => [
        "Hydrostatic model with a free surface" => "physics/hydrostatic_free_surface_model.md",
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
    "Generalized vertical coordinates" => "numerical_implementation/generalized_vertical_coordinates.md",
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

root = pkgdir(Oceananigans)
agents_src = joinpath(root, "AGENTS.md")
agents_dst = joinpath(root, "docs", "src", "developer_docs", "AGENTS.md")
cp(agents_src, agents_dst; force=true)

developer_pages = [
    "Contributor's guide" => "developer_docs/contributing.md",
    "Model interface" => "developer_docs/model_interface.md",
    "Implementing turbulence closures" => "developer_docs/turbulence_closures.md",
    "Rules for agent-assisted development" => "developer_docs/AGENTS.md",
]

pages = [
    "Home" => "index.md",
    "Quick start" => "quick_start.md",
    "Examples" => example_pages,
    "Grids" => "grids.md",
    "Fields" => "fields.md",
    "Operations" => "operations.md",
    # TODO:
    #   - Develop the following tutorials on reductions and post-processing
    #   - Refactor the model setup pages and make them more tutorial-like.
    # "Averages, integrals, and cumulative integrals" => "reductions_and_accumulations.md",
    # "FieldTimeSeries and post-processing" => field_time_series.md,
    "Models" => model_pages,
    "Simulations" => simulation_pages,
    "Physics" => physics_pages,
    "Numerical implementation" => numerical_pages,
    "Simulation tips" => "simulation_tips.md",
    "For developers" => developer_pages,
    "Gallery" => "gallery.md",
    "References" => "references.md",
    "Appendix" => appendix_pages
]

#####
##### Build and deploy docs
#####

format = Documenter.HTML(collapselevel = 1,
                         canonical = "https://clima.github.io/OceananigansDocumentation/stable/",
                         mathengine = MathJax3(),
                         size_threshold = 2^20,
                         assets = String["assets/citations.css"])

DocMeta.setdocmeta!(Oceananigans, :DocTestSetup, :(using Oceananigans); recursive=true)

modules = Module[]
OceananigansNCDatasetsExt = isdefined(Base, :get_extension) ? Base.get_extension(Oceananigans, :OceananigansNCDatasetsExt) : Oceananigans.OceananigansNCDatasetsExt
OceananigansXESMFExt = isdefined(Base, :get_extension) ? Base.get_extension(Oceananigans, :OceananigansXESMFExt) : Oceananigans.OceananigansXESMFExt

for m in [Oceananigans, XESMF, OceananigansNCDatasetsExt, OceananigansXESMFExt]
    if !isnothing(m)
        push!(modules, m)
    end
end

makedocs(; sitename = "Oceananigans.jl",
         authors = "Climate Modeling Alliance and contributors",
         format, pages, modules,
         plugins = [bib],
         warnonly = [:cross_references],
         doctestfilters = [
             r"┌ Warning:.*",  # remove standard warning lines
             r"└ @ .*",        # remove the source location of warnings
         ],
         clean = true,
         linkcheck = true,
         linkcheck_ignore = [
            r"jstor\.org",
            r"^https://github\.com/.*?/blob/",
         ],
         draft = false,        # set to true to speed things up
         doctest = true,       # set to false to speed things up
         checkdocs = :exports, # set to :none to speed things up
         )

"""
    recursive_find(directory, pattern)

Return list of filepaths within `directory` that contains the `pattern::Regex`.
"""
function recursive_find(directory, pattern)
    mapreduce(vcat, walkdir(directory)) do (root, dirs, filenames)
        matched_filenames = filter(contains(pattern), filenames)
        map(filename -> joinpath(root, filename), matched_filenames)
    end
end

@info "Cleaning up temporary .jld2 and .nc output created by doctests or literated examples..."

for pattern in [r"\.jld2", r"\.nc"]
    filenames = recursive_find(@__DIR__, pattern)

    for filename in filenames
        rm(filename)
    end
end

deploydocs(repo = "github.com/CliMA/OceananigansDocumentation.git",
           versions = ["stable" => "v^", "dev" => "dev", "v#.#.#"],
           forcepush = true,
           push_preview = true,
           devbranch = "main")
