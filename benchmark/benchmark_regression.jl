push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using PkgBenchmark
using Oceananigans
using Benchmarks

baseline = BenchmarkConfig(id="master")
script = joinpath(@__DIR__, "benchmarkable_nonhydrostatic_model.jl")
resultfile = joinpath(@__DIR__, "regression_benchmarks.json")

judge(Oceananigans, baseline, script=script, resultfile=resultfile, verbose=true)
