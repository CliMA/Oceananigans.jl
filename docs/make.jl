using Distributed
Distributed.addprocs(2)

@everywhere begin
    using DocumenterVitepress
    using Documenter
    using DocumenterCitations
    using Literate
    using Printf

    using CUDA
    using CairoMakie # to avoid capturing precompilation output by Literate
    set_theme!(Theme(fontsize=20))
    CairoMakie.activate!(type = "png")

    using Oceananigans
    using Oceananigans.AbstractOperations
    using Oceananigans.Operators
    using Oceananigans.Diagnostics
    using Oceananigans.OutputWriters
    using Oceananigans.TimeSteppers
    using Oceananigans.TurbulenceClosures
    using Oceananigans.BoundaryConditions: Flux, Value, Gradient, NormalFlow

    using NCDatasets
    using Zarr

    bib_filepath = joinpath(dirname(@__FILE__), "oceananigans.bib")
    bib = CitationBibliography(bib_filepath, style=:authoryear)

    #####
    ##### Generate examples
    #####

    const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
    const OUTPUT_DIR   = joinpath(@__DIR__, "src/literated")

    struct Example
        title :: String
        basename :: String
        build_always :: Bool # set to false for expensive examples built only on main, tags, or labeled PRs
    end

    # The examples that take longer to run should be first. This ensures that the
    # docs built with extra workers is as efficient as possible.
    all_examples = [
        Example("Global wind-driven gyres",               "global_wind_driven_gyres",           false),
        Example("Single column parameter estimation",     "single_column_parameter_estimation", false),
        Example("Ocean wind mixing and convection",       "ocean_wind_mixing_and_convection",   true),
        Example("Shallow water Bickley jet",              "shallow_water_Bickley_jet",          true),
        Example("Spherical baroclinic instability",       "spherical_baroclinic_instability",   false),
        Example("Barotropic jet and Double Drake on a Yin-Yang grid", "yin_yang_grid",          false),
        Example("Polar vortex crystal",                   "polar_vortex_crystal",               false),
        Example("Hydrostatic lock exchange with CATKE",   "hydrostatic_lock_exchange",          true),
        Example("Internal tide by a seamount",            "internal_tide",                      true),
        Example("Langmuir turbulence",                    "langmuir_turbulence",                false),
        Example("Kelvin-Helmholtz instability",           "kelvin_helmholtz_instability",       true),
        Example("Horizontal convection",                  "horizontal_convection",              true),
        Example("Baroclinic adjustment",                  "baroclinic_adjustment",              false),
        Example("Tilted bottom boundary layer",           "tilted_bottom_boundary_layer",       true),
        Example("Convecting plankton",                    "convecting_plankton",                true),
        Example("Two-dimensional turbulence",             "two_dimensional_turbulence",         true),
        Example("One-dimensional diffusion",              "one_dimensional_diffusion",          true),
        Example("Internal wave",                          "internal_wave",                      true),
    ]

    build_all_examples = get(ENV, "OCEANANIGANS_BUILD_ALL_EXAMPLES", "false") == "true"
    examples = filter(ex -> ex.build_always || build_all_examples, all_examples)
    example_scripts = [ex.basename * ".jl" for ex in examples]
end

# DocumenterVitepress 0.3.4 emits an ordered list directly after the preceding
# paragraph, without the blank line that VitePress requires to begin a list.
# This more-specific rendering method preserves native Markdown ordered lists.
function DocumenterVitepress.render(io::IO,
                                    mime::MIME"text/plain",
                                    node::DocumenterVitepress.MarkdownAST.Node,
                                    list::DocumenterVitepress.MarkdownAST.List,
                                    page::Documenter.Page,
                                    doc::Documenter.Document;
                                    kwargs...)
    println(io)

    item_number = 0
    bullet() = list.type === :ordered ? "$(item_number += 1). " : "- "
    item_buffer = IOBuffer()

    for item in node.children
        DocumenterVitepress.render(item_buffer, mime, item, item.children, page, doc;
                                   prenewline=false, kwargs...)
        lines = split(String(take!(item_buffer)), '\n')
        lines[2:end] .= "  " .* lines[2:end]
        print(io, bullet())
        println.((io,), lines)
    end

    return nothing
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

skipped_examples = setdiff(all_examples, examples)
if !isempty(skipped_examples)
    @info "Skipping expensive examples (set OCEANANIGANS_BUILD_ALL_EXAMPLES=true to build them): " *
          join([ex.basename for ex in skipped_examples], ", ")
end

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

# Pages are listed from simplest to most complex
page_order = ["one_dimensional_diffusion", "two_dimensional_turbulence", "internal_wave",
              "internal_tide", "convecting_plankton", "ocean_wind_mixing_and_convection",
              "langmuir_turbulence", "baroclinic_adjustment", "kelvin_helmholtz_instability",
              "hydrostatic_lock_exchange", "shallow_water_Bickley_jet", "horizontal_convection",
              "tilted_bottom_boundary_layer", "spherical_baroclinic_instability",
              "polar_vortex_crystal", "global_wind_driven_gyres", "yin_yang_grid",
              "single_column_parameter_estimation"]

sorted_examples = sort(examples, by = ex -> findfirst(==(ex.basename), page_order))
example_pages = [ex.title => joinpath("literated", ex.basename * ".md") for ex in sorted_examples]

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
    "Manual" => [
        "Quick Start" => "quick_start.md",
        "Units" => "units.md",
        "Grids" => "grids.md",
        "Immersed boundaries" => "immersed_boundaries.md",
        "Fields" => "fields.md",
        "Operations" => "operations.md",
        "Simulation Tips" => "simulation_tips.md",
        # Future tutorials:
        # "Reductions & Accumulations" => "reductions_and_accumulations.md",
        "FieldTimeSeries & Post-Processing" => "field_time_series.md",
        "Examples" => example_pages,
        "Gallery" => "gallery.md",
    ],
    "Workflows" => [
        "Models" => model_pages,
        "Simulations" => simulation_pages
    ],
    "Concepts" => [
        "Physics" => physics_pages,
        "Numerical Implementation" => numerical_pages
    ],
    "Developer" => developer_pages,
    "Resources" => [
        "References" => "references.md",
        "Appendix" => appendix_pages
    ]
]

#####
##### Build and deploy docs
#####

deploy_config = Documenter.auto_detect_deploy_system()
deploy_decision = Documenter.deploy_folder(
    deploy_config; repo="github.com/CliMA/OceananigansDocumentation.git",
    devbranch="main", devurl="dev", push_preview=true
)

format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/CliMA/Oceananigans.jl.git",
        devbranch = "main",
        devurl = "dev",
        deploy_url = "./OceananigansDocumentation/",
        deploy_decision,
        keep = :patch, # keep all versions of docs
    )

modules = Module[]
OceananigansNCDatasetsExt = isdefined(Base, :get_extension) ? Base.get_extension(Oceananigans, :OceananigansNCDatasetsExt) : Oceananigans.OceananigansNCDatasetsExt
OceananigansMakieExt = isdefined(Base, :get_extension) ? Base.get_extension(Oceananigans, :OceananigansMakieExt) : Oceananigans.OceananigansMakieExt

for m in [Oceananigans, OceananigansNCDatasetsExt, OceananigansMakieExt]
    if !isnothing(m)
        push!(modules, m)
    end
end

makedocs(; sitename = "Oceananigans.jl",
         authors = "Climate Modeling Alliance and contributors",
         format, pages, modules,
         plugins = [bib],
         warnonly = [:cross_references],
         clean = true,
         linkcheck = true,
         linkcheck_ignore = [
             r"jstor\.org",
             r"^https://github\.com/.*?/blob/",
             r"https://clima\.caltech\.edu/?",
         ],
         draft = false,        # set to true to speed things up
         doctest = false,      # doctests run in test/doctests/doctests.jl
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

@info "Cleaning up temporary .jld2 and .nc output created by literated examples..."

for pattern in [r"\.jld2", r"\.nc"]
    filenames = recursive_find(@__DIR__, pattern)

    for filename in filenames
        rm(filename)
    end
end

DocumenterVitepress.deploydocs(repo = "github.com/CliMA/Oceananigans.jl.git",
                               deploy_repo = "github.com/CliMA/OceananigansDocumentation.git",
                               target = "build",
                               branch = "gh-pages",
                               forcepush = true,
                               push_preview = true,
                               devbranch = "main")
