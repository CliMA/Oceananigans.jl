module Benchmarks

export @sync_gpu,
       Slab, Pencil,
       print_system_info,
       run_benchmarks,
       benchmarks_dataframe,
       benchmarks_pretty_table,
       gpu_speedups_suite,
       speedups_suite,
       speedups_dataframe

using Logging
using BenchmarkTools
using DataFrames
using PrettyTables
using CUDA

using BenchmarkTools: prettytime, prettymemory
using Oceananigans: OceananigansLogger
using Oceananigans.Architectures: CPU, GPU
using Oceananigans.Utils: oceananigans_versioninfo, versioninfo_with_gpu

abstract type AbstractDomainDecomposition end

struct Slab <: AbstractDomainDecomposition end
struct Pencil <: AbstractDomainDecomposition end

function __init__()
    Logging.global_logger(OceananigansLogger())
end

macro sync_gpu(expr)
    return CUDA.has_cuda() ? :($(esc(CUDA.@sync expr))) : :($(esc(expr)))
end

function print_system_info()
    println()
    println(oceananigans_versioninfo())
    println(versioninfo_with_gpu())
    println()
    return nothing
end

function run_benchmarks(benchmark_fun; kwargs...)
    keys = [p.first for p in kwargs]
    vals = [p.second for p in kwargs]

    cases = Iterators.product(vals...)
    n_cases = length(cases)

    tags = string.(keys)
    suite = BenchmarkGroup(tags)
    for (n, case) in enumerate(cases)
        @info "Benchmarking $n/$n_cases: $case..."
        suite[case] = benchmark_fun(case...)
        GC.gc()
        GC.gc(true)
        GC.gc()
    end
    return suite
end

function benchmarks_dataframe(suite)
    names = Tuple(Symbol(tag) for tag in suite.tags)
    df_names = (names..., :min, :median, :mean, :max, :memory, :allocs, :samples)
    empty_cols = Tuple([] for k in df_names)
    df = DataFrame(; NamedTuple{df_names}(empty_cols)...)

    for case in keys(suite)
        trial = suite[case]
        entry = NamedTuple{names}(case) |> pairs |> Dict{Any,Any}

        entry[:min] = minimum(trial.times) |> prettytime
        entry[:median] = median(trial.times) |> prettytime
        entry[:mean] = mean(trial.times) |> prettytime
        entry[:max] = maximum(trial.times) |> prettytime
        entry[:memory] = prettymemory(trial.memory)
        entry[:allocs] = trial.allocs
        entry[:samples] = length(trial)

        push!(df, entry)
    end

    return df
end

function benchmarks_pretty_table(df; title="")
    header = propertynames(df) .|> String
    pretty_table(df, header, nosubheader=true, title=title, title_alignment=:c,
                 title_autowrap = true, title_same_width_as_table = true)

    html_filename = replace(title, ' ' => '_') * ".html"
    @info "Writing $html_filename..."
    open(html_filename, "w") do io
        html_table = pretty_table(String, df, header, nosubheader=true,
                                  title=title, title_alignment=:c,
                                  backend=:html, tf=tf_html_simple)
        write(io, html_table)
    end

    return nothing
end

is_arch_type(e) = e == CPU || e == GPU
cpu_case(case) = Tuple(is_arch_type(e) ? CPU : e for e in case)
gpu_case(case) = Tuple(is_arch_type(e) ? GPU : e for e in case)

function gpu_speedups_suite(suite)
    tags = filter(e -> !occursin("arch", lowercase(e)), suite.tags)
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

function speedups_suite(suite; base_case)
    suite_speedup = BenchmarkGroup(suite.tags)

    for case in keys(suite)
        suite_speedup[case] = ratio(median(suite[case]), median(suite[base_case]))
    end

    return suite_speedup
end

function speedups_dataframe(suite; slowdown=false, efficiency=nothing, base_case=nothing, key2rank=identity)
    names = Tuple(Symbol(tag) for tag in suite.tags)
    speed_type = slowdown ? :slowdown : :speedup
    df_names = isnothing(efficiency) ? (names..., speed_type, :memory, :allocs) : (names..., speed_type, :efficiency, :memory, :allocs)
    empty_cols = Tuple([] for k in df_names)
    df = DataFrame(; NamedTuple{df_names}(empty_cols)...)

    for case in keys(suite)
        trial_ratio = suite[case]
        entry = NamedTuple{names}(case) |> pairs |> Dict{Any,Any}

        entry[speed_type] = slowdown ? trial_ratio.time :  1/trial_ratio.time
        entry[:memory] = trial_ratio.memory
        entry[:allocs] = trial_ratio.allocs

        if efficiency == :strong
            R = key2rank(case)
            t₁ = suite[base_case].time
            tₙ = suite[case].time
            entry[:efficiency] = t₁ / (R * tₙ)
        end

        if efficiency == :weak
            R = key2rank(case)
            t₁ = suite[base_case].time
            tₙ = suite[case].time
            entry[:efficiency] = t₁ / tₙ
        end

        push!(df, entry)
    end

    return df
end

end # module
