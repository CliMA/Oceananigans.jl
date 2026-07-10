using Distributed
Distributed.addprocs(2)

@everywhere begin
    using DocumenterVitepress
    using Documenter
    using DocumenterCitations
    using Literate
    using Printf
    using Markdown

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

    bib_filepath = joinpath(dirname(@__FILE__), "oceananigans.bib")
    bib = CitationBibliography(bib_filepath, style=:authoryear)

    #####
    ##### Generate examples
    #####

    const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
    const OUTPUT_DIR   = joinpath(@__DIR__, "src/literated")

    # The examples that take longer to run should be first. This ensures that the
    # docs built with extra workers is as efficient as possible.
    example_scripts = [
        "ocean_wind_mixing_and_convection.jl",
        "shallow_water_Bickley_jet.jl",
        "spherical_baroclinic_instability.jl",
        "polar_vortex_crystal.jl",
        "hydrostatic_lock_exchange.jl",
        "internal_tide.jl",
        "langmuir_turbulence.jl",
        "kelvin_helmholtz_instability.jl",
        "horizontal_convection.jl",
        "baroclinic_adjustment.jl",
        "tilted_bottom_boundary_layer.jl",
        "convecting_plankton.jl",
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
    "Spherical baroclinic instability"      => "literated/spherical_baroclinic_instability.md",
    "Polar vortex crystal"                  => "literated/polar_vortex_crystal.md"
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
    "Manual" => [
        "Quick Start" => "quick_start.md",
        "Units" => "units.md",
        "Grids" => "grids.md",
        "Fields" => "fields.md",
        "Operations" => "operations.md",
        "Simulation Tips" => "simulation_tips.md",
        # Future tutorials:
        # "Reductions & Accumulations" => "reductions_and_accumulations.md",
        # "FieldTimeSeries & Post-Processing" => field_time_series.md
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

function paper_counts_by_year(index_path)
    text = read(index_path, String)
    papers_heading = "## Papers and preprints using Oceananigans"
    papers_start = findfirst(papers_heading, text)
    isnothing(papers_start) && error("Could not find publications section in $(index_path).")

    papers_section = text[first(papers_start):end]
    year_counts = Dict{Int, Int}()

    for line in eachline(IOBuffer(papers_section))
        startswith(line, "1. ") || continue
        year_match = match(r"\((\d{4})\)", line)
        isnothing(year_match) && continue
        year = parse(Int, year_match.captures[1])
        year_counts[year] = get(year_counts, year, 0) + 1
    end

    return sort(collect(year_counts); by=first)
end

function render_oceananigans_paper_timeline()
    index_path = joinpath(pkgdir(Oceananigans), "docs", "src", "index.md")
    counts = paper_counts_by_year(index_path)
    total_papers = sum(last, counts)
    maximum_count = maximum(last, counts)

    html = IOBuffer()

    println(html, "<style>")
    println(html, ".oceananigans-paper-timeline {")
    println(html, "  margin: 1.5rem 0 2rem 0;")
    println(html, "  padding: 1rem 1.25rem 0.75rem 1.25rem;")
    println(html, "  border: 1px solid var(--vp-c-divider);")
    println(html, "  border-radius: 14px;")
    println(html, "  background: linear-gradient(180deg, color-mix(in srgb, var(--vp-c-brand-1) 8%, transparent), transparent 65%);")
    println(html, "}")
    println(html, ".oceananigans-paper-timeline-title {")
    println(html, "  margin: 0 0 0.2rem 0;")
    println(html, "  font-size: 0.95rem;")
    println(html, "  font-weight: 600;")
    println(html, "  color: var(--vp-c-text-1);")
    println(html, "}")
    println(html, ".oceananigans-paper-timeline-subtitle {")
    println(html, "  margin: 0 0 0.8rem 0;")
    println(html, "  font-size: 0.8rem;")
    println(html, "  color: var(--vp-c-text-2);")
    println(html, "}")
    println(html, ".oceananigans-paper-bars {")
    println(html, "  display: flex;")
    println(html, "  align-items: end;")
    println(html, "  gap: 0.7rem;")
    println(html, "  min-height: 12rem;")
    println(html, "}")
    println(html, ".oceananigans-paper-bar-group {")
    println(html, "  display: flex;")
    println(html, "  flex: 1 1 0;")
    println(html, "  flex-direction: column;")
    println(html, "  align-items: center;")
    println(html, "  gap: 0.35rem;")
    println(html, "}")
    println(html, ".oceananigans-paper-count {")
    println(html, "  font-size: 0.75rem;")
    println(html, "  color: var(--vp-c-text-2);")
    println(html, "}")
    println(html, ".oceananigans-paper-bar {")
    println(html, "  width: 100%;")
    println(html, "  max-width: 2.6rem;")
    println(html, "  min-height: 0.35rem;")
    println(html, "  border-radius: 6px 6px 0 0;")
    println(html, "  background: linear-gradient(180deg, var(--vp-c-brand-1), var(--vp-c-brand-3));")
    println(html, "  box-shadow: 0 0 0 1px color-mix(in srgb, var(--vp-c-brand-1) 18%, transparent);")
    println(html, "}")
    println(html, ".oceananigans-paper-year {")
    println(html, "  font-size: 0.75rem;")
    println(html, "  color: var(--vp-c-text-2);")
    println(html, "}")
    println(html, "</style>")
    println(html)
    println(html, "<div class=\"oceananigans-paper-timeline\">")
    println(html, "  <p class=\"oceananigans-paper-timeline-title\">Papers per year</p>")
    println(html, "  <p class=\"oceananigans-paper-timeline-subtitle\">$(total_papers) verified papers and preprints currently listed</p>")
    println(html, "  <div class=\"oceananigans-paper-bars\">")

    for (year, count) in counts
        height = round(100 * count / maximum_count; digits=3)
        println(html, "    <div class=\"oceananigans-paper-bar-group\">")
        println(html, "      <div class=\"oceananigans-paper-count\">$(count)</div>")
        println(html, "      <div class=\"oceananigans-paper-bar\" style=\"height: $(height)%;\"></div>")
        println(html, "      <div class=\"oceananigans-paper-year\">$(year)</div>")
        println(html, "    </div>")
    end

    println(html, "  </div>")
    println(html, "</div>")

    return Markdown.MD(Documenter.RawNode(:html, String(take!(html))))
end

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

DocMeta.setdocmeta!(Oceananigans, :DocTestSetup, :(using Oceananigans); recursive=true)

modules = Module[]
OceananigansNCDatasetsExt = isdefined(Base, :get_extension) ? Base.get_extension(Oceananigans, :OceananigansNCDatasetsExt) : Oceananigans.OceananigansNCDatasetsExt

for m in [Oceananigans, OceananigansNCDatasetsExt]
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
             r"https://clima\.caltech\.edu/?",
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

DocumenterVitepress.deploydocs(repo = "github.com/CliMA/Oceananigans.jl.git",
                               deploy_repo = "github.com/CliMA/OceananigansDocumentation.git",
                               target = "build",
                               branch = "gh-pages",
                               forcepush = true,
                               push_preview = true,
                               devbranch = "main")
