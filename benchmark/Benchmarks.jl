module Benchmarks

export @sync_gpu

using BenchmarkTools
using DataFrames
using PrettyTables

using BenchmarkTools: prettytime, prettymemory

macro sync_gpu(expr)
    return has_cuda() :($(esc(CUDA.@sync expr))) : :(esc(expr))
end

function run_benchmark_suite(benchmark_fun; kwargs...)
    keys = [p.first for p in kwargs]
    vals = [p.second for p in kwargs]

    cases = Iterators.product(vals...)
    n_cases = length(cases)

    tags = string.(keys)
    suite = BenchmarkGroup(tags)
    for case in cases
        @info "Benchmarking $case..."
        suite[case] = benchmark_fun(case...)
    end
    return suite
end

end