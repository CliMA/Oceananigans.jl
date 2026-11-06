using Documenter
using Oceananigans

# Doctest output prints types relative to `Main`, so `Main` needs the names that docs/make.jl brings in
@eval Main begin
    using CUDA
    using CairoMakie
    using NCDatasets
    using Zarr

    using Oceananigans
    using Oceananigans.AbstractOperations
    using Oceananigans.Operators
    using Oceananigans.Diagnostics
    using Oceananigans.OutputWriters
    using Oceananigans.TimeSteppers
    using Oceananigans.TurbulenceClosures
    using Oceananigans.BoundaryConditions: Flux, Value, Gradient, NormalFlow
end

DocMeta.setdocmeta!(Oceananigans, :DocTestSetup, :(using Oceananigans); recursive=true)

modules = [Oceananigans,
           Base.get_extension(Oceananigans, :OceananigansMakieExt),
           Base.get_extension(Oceananigans, :OceananigansNCDatasetsExt)]

doctestfilters = [r"┌ Warning:.*",  # remove standard warning lines
                  r"└ @ .*"]        # remove the source location of warnings

manual = joinpath(@__DIR__, "..", "..", "docs", "src")

doctest(manual, modules; doctestfilters, testset="Doctests")
