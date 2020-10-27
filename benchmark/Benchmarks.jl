module Benchmarks

export @sync_gpu,
       run_benchmark_suite,
       benchmark_suite_to_dataframe,
       summarize_benchmark_suite,
       gpu_speedup_suite,
       speedup_suite,
       speedup_suite_to_dataframe

using BenchmarkTools
using DataFrames
using PrettyTables
using CUDA

using BenchmarkTools: prettytime, prettymemory
using Oceananigans.Architectures: CPU, GPU

macro sync_gpu(expr)
    return CUDA.has_cuda() ? :($(esc(CUDA.@sync expr))) : :($(esc(expr)))
end

function run_benchmarks(benchmark_fun; kwargs...)
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

function benchmarks_dataframe(suite)
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

function benchmarks_pretty_table(df; title="")
    header = propertynames(df) .|> String
    pretty_table(df, header, title=title, nosubheader=true)
    return nothing
end

is_arch_type(e) = e == CPU || e == GPU
cpu_case(case) = Tuple(is_arch_type(e) ? CPU : e for e in case)
gpu_case(case) = Tuple(is_arch_type(e) ? GPU : e for e in case)

function gpu_speedups_suite(suite)
    tags = filter(e -> e != "Archs", suite.tags)
    suite_speedup = BenchmarkGroup(tags)

    for case in keys(suite)
        case_cpu = cpu_case(case)
        case_gpu = gpu_case(case)
        case_speedup = filter(!is_arch_type, case_cpu)

        if case_speedup ∉ keys(suite_speedup)
            suite_speedup[case_speedup] = ratio(median(suite[case_gpu]), median(suite[case_cpu]))
        end
    end

    return suite_speedup
end

function speedups_suite(suite, base_case)
    suite_speedup = BenchmarkGroup(suite.tags)

    for case in keys(suite)
        suite_speedup[case] = ratio(median(suite[case]), median(suite[base_case]))
    end

    return suite_speedup
end

function speedups_dataframe(suite)
    names = Tuple(Symbol(tag) for tag in suite.tags)
    df_names = (names..., :speedup, :memory, :allocs)
    empty_cols = Tuple([] for k in df_names)
    df = DataFrame(; NamedTuple{df_names}(empty_cols)...)
    
    for case in keys(suite)
        trial_ratio = suite[case]
        entry = NamedTuple{names}(case) |> pairs |> Dict

        entry[:speedup] = 1/trial_ratio.time
        entry[:memory] = trial_ratio.memory
        entry[:allocs] = trial_ratio.allocs

        push!(df, entry)
    end

    return df
end

end