include("dependencies_for_runtests.jl")

using Oceananigans.Utils: TabulatedFunction, tabulate

@testset "Utils" begin
    @info "Testing utils..."

    @testset "prettytime" begin
        @test prettytime(0) == "0 seconds"
        @test prettytime(35e-15) == "3.500e-14 seconds"

        @test prettytime(1e-9) == "1 ns"
        @test prettytime(1e-6) == "1 μs"
        @test prettytime(1e-3) == "1 ms"

        @test prettytime(second) == "1 second"
        @test prettytime(minute) == "1 minute"
        @test prettytime(hour) == "1 hour"
        @test prettytime(day) == "1 day"

        @test prettytime(2second) == "2 seconds"
        @test prettytime(4minute) == "4 minutes"
        @test prettytime(8hour) == "8 hours"
        @test prettytime(16day) == "16 days"

        @test prettytime(13.7seconds) == "13.700 seconds"
        @test prettytime(6.666minutes) == "6.666 minutes"
        @test prettytime(1.234hour) == "1.234 hours"
        @test prettytime(40.5days) == "40.500 days"
    end

    @testset "prettysummary" begin
        f4905(x::Integer) = 0
        f4905(x::String) = ""
        f4905(x::AbstractFloat) = 0.0

        @test prettysummary(f4905, false) == "f4905"
        @test prettysummary(f4905, true) == "f4905 (generic function with 3 methods)"
        @test prettysummary(prettysummary, false) == "prettysummary"
        @test contains(prettysummary(prettysummary, true), r"^prettysummary \(generic function with \d+ methods\)$")
        @test contains(prettysummary(x -> x, true), r"^#\d+ \(generic function with 1 method\)$")
    end

    @testset "TabulatedFunction" begin
        @info "  Testing TabulatedFunction..."

        # Test basic construction and evaluation
        f = TabulatedFunction(sin; range=(0, 2π), points=1000)
        @test f isa TabulatedFunction
        @test abs(f(π/4) - sin(π/4)) < 0.001
        @test abs(f(π/2) - sin(π/2)) < 0.001
        @test abs(f(π) - sin(π)) < 0.001

        # Test tabulate alias
        g = tabulate(x -> x^2; range=(-5, 5), points=500)
        @test g isa TabulatedFunction
        @test abs(g(2.0) - 4.0) < 0.01
        @test abs(g(-3.0) - 9.0) < 0.01

        # Test clamping at boundaries
        h = TabulatedFunction(identity; range=(0, 1), points=100)
        @test h(-0.5) ≈ 0.0  # Clamped to x_min
        @test h(1.5) ≈ 1.0   # Clamped to x_max

        # Test with Float32
        f32 = TabulatedFunction(cos, CPU(), Float32; range=(0, π), points=100)
        @test eltype(f32.table) == Float32
        @test abs(f32(π/2) - cos(Float32(π/2))) < 0.01

        # Test expensive function (what it's designed for)
        expensive_func(x) = log(1 + exp(x)) + sqrt(abs(x))
        t = TabulatedFunction(expensive_func; range=(-10, 10), points=10000)
        @test abs(t(0.0) - expensive_func(0.0)) < 0.1  # sqrt has a singularity at 0
        @test abs(t(5.0) - expensive_func(5.0)) < 0.01

        # Test summary/show
        @test contains(summary(f), "TabulatedFunction")
        @test contains(summary(f), "1000 points")
    end
end
