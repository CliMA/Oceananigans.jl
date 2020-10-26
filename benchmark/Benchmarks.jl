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

function benchmark_suite_to_dataframe(suite)
    names = Tuple(Symbol(tag) for tag in suite.tags)
    df_names = (names..., :min, :median, :mean, :max, :memory, :allocs)
    empty_cols = Tuple([] for k in df_names)
    df = DataFrame(; NamedTuple{df_names}(empty_cols)...)
    
    for case in keys(suite)
        trial = suite[case]
        entry = NamedTuple{names}(case) |> pairs |> Dict
       
        entry[:min] = minimum(trial.times) |> prettytime
        entry[:median] = median(trial.times) |> prettytime
        entry[:mean] = mean(trial.times) |> prettytime
        entry[:max] = maximum(trial.times) |> prettytime
        entry[:memory] = prettymemory(trial.memory)
        entry[:allocs] = trial.allocs

        push!(df, entry)
    end

    return df
end

function summarize_benchmark_suite(df)
    header = propertynames(df) .|> String
    pretty_table(df, header, title="Incompressible model benchmarks", nosubheader=true)
    return nothing
end

end