using BenchmarkTools

const BENCHMARKS_DIR = "../benchmark/"
cp(joinpath(BENCHMARKS_DIR, "benchmark_utils.jl"), joinpath(@__DIR__, "benchmark_utils.jl"), force=true)

function run_benchmark(replace_strings, benchmark_name, module_suffix="")
    benchmark_filepath = joinpath(BENCHMARKS_DIR, "benchmark_" * benchmark_name * ".jl")
    txt = read(benchmark_filepath, String)

    for strs in replace_strings
        txt = replace(txt, strs[1] => strs[2])
    end

    test_script_filepath = benchmark_name * "_benchmark_test.jl"

    open(test_script_filepath, "w") do f
        write(f, "module Test_$benchmark_name" * "_$module_suffix\n")
        write(f, txt)
        write(f, "\nend # module")
    end

    try
        include(test_script_filepath)
    catch err
        @error sprint(showerror, err)
        rm(test_script_filepath)
        return false
    end

    rm(test_script_filepath)
    return true
end

@testset "Performance benchmarks" begin
    @info "Testing performance benchmarks..."

    @testset "Performance benchmark scripts" begin
        @info "  Testing performance benchmark scripts..."

        @testset "Static ocean benchmark" begin
            @info "    Running static ocean benchmarks..."

            replace_strings = [
                ("Ns = [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)]",
                 "Ns = [(16, 16, 16)]")
            ]

            @test run_benchmark(replace_strings, "static_ocean")
        end

        @testset "Channel benchmark" begin
            @info "    Running channel benchmarks..."

            replace_strings = [
                ("Ns = [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)]",
                 "Ns = [(16, 16, 16)]")
            ]

            @test run_benchmark(replace_strings, "channel")
        end

        @testset "Turbulence closures benchmark" begin
            @info "    Running turbulence closures benchmark..."

            replace_strings = [
                ("Ns = [(32, 32, 32), (128, 128, 128)]",
                 "Ns = [(16, 16, 16)]")
            ]

            @test run_benchmark(replace_strings, "turbulence_closures")
        end

        @testset "Tracers benchmark" begin
            @info "    Running tracers benchmark..."

            replace_strings = [
                ("(32, 32, 32)", "(16, 16, 16)"),
                ("(256, 256, 256)", "(16, 16, 16)"),
                ("test_cases = [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 3), (2, 5), (2, 10)]",
                 "test_cases = [(0, 0), (2, 0), (2, 3)]")
            ]

            @test run_benchmark(replace_strings, "tracers")
        end
    end

    @testset "Selected performance benchmarks" begin
        @info "  Running selected performance benchmarks..."

        for arch in archs
            sizes = [(16, 16, 16), (32, 32, 32)]
            for sz in sizes
                grid = RegularCartesianGrid(size=sz, length=(1, 1, 1))
                model = IncompressibleModel(architecture=arch, grid=grid)

                @info "Benchmarking [$arch, Float64, $sz]..."
                b = @benchmark time_step!($model, 1)
                display(b)
                println()

                @test model isa IncompressibleModel
            end
        end
    end
end

rm(joinpath(@__DIR__, "benchmark_utils.jl"))
