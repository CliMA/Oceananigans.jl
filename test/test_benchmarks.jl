using BenchmarkTools

const BENCHMARKS_DIR = "../benchmark/"
cp(joinpath(BENCHMARKS_DIR, "benchmark_utils.jl"), joinpath(@__DIR__, "benchmark_utils.jl"), force=true)

benchmark_filepath(benchmark_name, benchmarks_dir=BENCHMARKS_DIR) =
    joinpath(benchmarks_dir, "benchmark_" * benchmark_name * ".jl")

@testset "Performance benchmarks" begin
    @info "Testing performance benchmarks..."

    @testset "Performance benchmark scripts" begin
        @info "  Testing performance benchmark scripts..."

        @testset "Static ocean benchmark" begin
            @info "    Running static ocean benchmarks..."

            replace_strings = [
                ("Ns = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)]",
                 "Ns = [(1, 1, 1)]")
            ]

            @test run_script(replace_strings, "static_ocean", benchmark_filepath("static_ocean"))
        end

        @testset "Channel benchmark" begin
            @info "    Running channel benchmarks..."

            replace_strings = [
                ("Ns = [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)]",
                 "Ns = [(1, 1, 1)]")
            ]

            @test run_script(replace_strings, "channel", benchmark_filepath("channel"))
        end

        @testset "Turbulence closures benchmark" begin
            @info "    Running turbulence closures benchmark..."

            replace_strings = [
                ("Ns = [(32, 32, 32), (256, 256, 128)]",
                 "Ns = [(1, 1, 1)]")
            ]

            @test run_script(replace_strings, "turbulence_closures", benchmark_filepath("turbulence_closures"))
        end

        @testset "Tracers benchmark" begin
            @info "    Running tracers benchmark..."

            replace_strings = [
                ("(32, 32, 32)", "(1, 1, 1)"),
                ("(256, 256, 128)", "(1, 1, 1)"),
                ("test_cases = [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 3), (2, 5), (2, 10)]",
                 "test_cases = [(0, 0), (2, 0), (2, 3)]")
            ]

            @test run_script(replace_strings, "tracers", benchmark_filepath("tracers"))
        end
    end

    @testset "Selected performance benchmarks" begin
        @info "  Running selected performance benchmarks..."

        for arch in archs
            sizes = [(16, 16, 16), (32, 32, 32)]
            for sz in sizes
                grid = RegularCartesianGrid(size=sz, extent=(1, 1, 1))
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
