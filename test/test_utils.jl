include("dependencies_for_runtests.jl")

using Oceananigans.Utils: TabulatedFunction

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

        #####
        ##### 1D TabulatedFunction
        #####

        @testset "1D TabulatedFunction" begin
            # Test basic construction and evaluation
            f = TabulatedFunction(sin; range=(0, 2π), points=1000)
            @test f isa TabulatedFunction{1}
            @test abs(f(π/4) - sin(π/4)) < 0.001
            @test abs(f(π/2) - sin(π/2)) < 0.001
            @test abs(f(π) - sin(π)) < 0.001

            # Test with anonymous function
            g = TabulatedFunction(x -> x^2; range=(-5, 5), points=500)
            @test g isa TabulatedFunction{1}
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
            @test contains(summary(f), "TabulatedFunction{1}")
            @test contains(summary(f), "1000 points")

            # Test exact values at grid points
            f_lin = TabulatedFunction(identity; range=(0, 10), points=11)
            for i in 0:10
                @test f_lin(Float64(i)) ≈ Float64(i)
            end

            # Test internal structure
            @test length(f.range) == 1
            @test length(f.inverse_Δ) == 1
            @test f.range[1][1] ≈ 0.0
            @test f.range[1][2] ≈ 2π
        end

        #####
        ##### 2D TabulatedFunction (bilinear interpolation)
        #####

        @testset "2D TabulatedFunction" begin
            g2d(x, y) = sin(x) * cos(y)
            f2d = TabulatedFunction(g2d; range=((0, π), (0, 2π)), points=(100, 200))
            @test f2d isa TabulatedFunction{2}
            @test size(f2d.table) == (100, 200)

            # Test accuracy
            @test abs(f2d(π/4, π/4) - g2d(π/4, π/4)) < 0.01
            @test abs(f2d(π/2, π) - g2d(π/2, π)) < 0.01

            # Test with scalar points (broadcast to both dimensions)
            f2d_scalar = TabulatedFunction(g2d; range=((0, π), (0, 2π)), points=50)
            @test size(f2d_scalar.table) == (50, 50)

            # Test clamping in 2D
            h2d(x, y) = x + y
            t2d = TabulatedFunction(h2d; range=((0, 1), (0, 1)), points=100)
            @test t2d(-0.5, 0.5) ≈ 0.5  # x clamped to 0
            @test t2d(0.5, 1.5) ≈ 1.5   # y clamped to 1
            @test t2d(1.5, 1.5) ≈ 2.0   # both clamped

            # Test summary for 2D
            @test contains(summary(f2d), "TabulatedFunction{2}")
            @test contains(summary(f2d), "100×200")

            # Test exact values at grid points (linear function should be exact)
            f2d_lin = TabulatedFunction((x, y) -> x + y; range=((0, 1), (0, 1)), points=(11, 11))
            for i in 0:10, j in 0:10
                x, y = i/10, j/10
                @test f2d_lin(x, y) ≈ x + y atol=1e-10
            end

            # Test Float32 for 2D
            f2d_32 = TabulatedFunction(g2d, CPU(), Float32; range=((0, π), (0, 2π)), points=(50, 50))
            @test eltype(f2d_32.table) == Float32

            # Test internal structure
            @test length(f2d.range) == 2
            @test length(f2d.inverse_Δ) == 2
            @test f2d.range[1][1] ≈ 0.0
            @test f2d.range[1][2] ≈ π
            @test f2d.range[2][1] ≈ 0.0
            @test f2d.range[2][2] ≈ 2π

            # Test asymmetric points
            f2d_asym = TabulatedFunction(g2d; range=((0, 1), (0, 2)), points=(10, 50))
            @test size(f2d_asym.table) == (10, 50)
        end

        #####
        ##### 3D TabulatedFunction (trilinear interpolation)
        #####

        @testset "3D TabulatedFunction" begin
            g3d(x, y, z) = x^2 + y^2 + z^2
            f3d = TabulatedFunction(g3d; range=((-1, 1), (-1, 1), (-1, 1)), points=(20, 20, 20))
            @test f3d isa TabulatedFunction{3}
            @test size(f3d.table) == (20, 20, 20)

            # Test accuracy
            @test abs(f3d(0.0, 0.0, 0.0) - g3d(0.0, 0.0, 0.0)) < 0.01
            @test abs(f3d(0.5, 0.5, 0.5) - g3d(0.5, 0.5, 0.5)) < 0.05

            # Test with scalar points (broadcast to all dimensions)
            f3d_scalar = TabulatedFunction(g3d; range=((-1, 1), (-1, 1), (-1, 1)), points=15)
            @test size(f3d_scalar.table) == (15, 15, 15)

            # Test clamping in 3D
            h3d(x, y, z) = x * y * z
            t3d = TabulatedFunction(h3d; range=((0, 1), (0, 1), (0, 1)), points=50)
            @test t3d(0.5, 0.5, 0.5) ≈ 0.125 atol=0.01

            # Test clamping at all boundaries
            @test t3d(-1, 0.5, 0.5) ≈ 0.0 atol=0.01  # x clamped to 0
            @test t3d(0.5, -1, 0.5) ≈ 0.0 atol=0.01  # y clamped to 0
            @test t3d(0.5, 0.5, -1) ≈ 0.0 atol=0.01  # z clamped to 0
            @test t3d(2, 0.5, 0.5) ≈ 0.25 atol=0.01  # x clamped to 1
            @test t3d(0.5, 2, 0.5) ≈ 0.25 atol=0.01  # y clamped to 1
            @test t3d(0.5, 0.5, 2) ≈ 0.25 atol=0.01  # z clamped to 1

            # Test summary for 3D
            @test contains(summary(f3d), "TabulatedFunction{3}")
            @test contains(summary(f3d), "20×20×20")

            # Test exact values at grid points (linear function should be exact)
            f3d_lin = TabulatedFunction((x, y, z) -> x + y + z; range=((0, 1), (0, 1), (0, 1)), points=(11, 11, 11))
            for i in 0:10, j in 0:10, k in 0:10
                x, y, z = i/10, j/10, k/10
                @test f3d_lin(x, y, z) ≈ x + y + z atol=1e-10
            end

            # Test Float32 for 3D
            f3d_32 = TabulatedFunction(g3d, CPU(), Float32; range=((-1, 1), (-1, 1), (-1, 1)), points=10)
            @test eltype(f3d_32.table) == Float32

            # Test internal structure
            @test length(f3d.range) == 3
            @test length(f3d.inverse_Δ) == 3

            # Test asymmetric points
            f3d_asym = TabulatedFunction(g3d; range=((-1, 1), (-1, 1), (-1, 1)), points=(10, 20, 30))
            @test size(f3d_asym.table) == (10, 20, 30)
        end

        #####
        ##### Architecture and Adapt tests
        #####

        @testset "Architecture and Adapt" begin
            using Oceananigans.Architectures: on_architecture
            using Adapt

            # Test on_architecture for 1D
            f1 = TabulatedFunction(sin; range=(0, 2π), points=100)
            f1_cpu = on_architecture(CPU(), f1)
            @test f1_cpu.table isa Vector
            @test f1_cpu(π/2) ≈ f1(π/2)

            # Test on_architecture for 2D
            f2 = TabulatedFunction((x, y) -> x * y; range=((0, 1), (0, 1)), points=50)
            f2_cpu = on_architecture(CPU(), f2)
            @test f2_cpu.table isa Matrix
            @test f2_cpu(0.5, 0.5) ≈ f2(0.5, 0.5)

            # Test on_architecture for 3D
            f3 = TabulatedFunction((x, y, z) -> x + y + z; range=((0, 1), (0, 1), (0, 1)), points=20)
            f3_cpu = on_architecture(CPU(), f3)
            @test f3_cpu.table isa Array{Float64, 3}
            @test f3_cpu(0.5, 0.5, 0.5) ≈ f3(0.5, 0.5, 0.5)

            # Test Adapt.adapt_structure (used for GPU kernels)
            # When adapted, func should be replaced with nothing
            f1_adapted = Adapt.adapt_structure(nothing, f1)
            @test f1_adapted.func === nothing
            @test f1_adapted.range == f1.range
            @test f1_adapted.inverse_Δ == f1.inverse_Δ

            f2_adapted = Adapt.adapt_structure(nothing, f2)
            @test f2_adapted.func === nothing
            @test f2_adapted isa TabulatedFunction{2}

            f3_adapted = Adapt.adapt_structure(nothing, f3)
            @test f3_adapted.func === nothing
            @test f3_adapted isa TabulatedFunction{3}
        end

        #####
        ##### Edge cases and special values
        #####

        @testset "Edge cases" begin
            # Test with very small range
            f_small = TabulatedFunction(sin; range=(0, 1e-10), points=100)
            @test f_small(0.5e-10) isa Float64

            # Test evaluation at exact min/max
            f = TabulatedFunction(identity; range=(0, 1), points=101)
            @test f(0.0) ≈ 0.0
            @test f(1.0) ≈ 1.0

            # Test with negative range
            f_neg = TabulatedFunction(identity; range=(-10, -5), points=100)
            @test f_neg(-7.5) ≈ -7.5 atol=0.1

            # Test with 2 points (minimum for interpolation)
            f_min = TabulatedFunction(identity; range=(0, 1), points=2)
            @test f_min(0.0) ≈ 0.0
            @test f_min(1.0) ≈ 1.0
            @test f_min(0.5) ≈ 0.5

            # Test that show works with nothing func (after adapt)
            f_adapted = Adapt.adapt_structure(nothing, TabulatedFunction(sin; range=(0, 1)))
            @test contains(summary(f_adapted), "TabulatedFunction{1}")
            io = IOBuffer()
            show(io, f_adapted)
            @test length(String(take!(io))) > 0
        end

        #####
        ##### Type alias tests
        #####

        @testset "Type aliases" begin
            using Oceananigans.Utils: TabulatedFunction1D, TabulatedFunction2D, TabulatedFunction3D

            f1 = TabulatedFunction(sin; range=(0, 1))
            @test f1 isa TabulatedFunction1D

            f2 = TabulatedFunction((x, y) -> x + y; range=((0, 1), (0, 1)))
            @test f2 isa TabulatedFunction2D

            f3 = TabulatedFunction((x, y, z) -> x + y + z; range=((0, 1), (0, 1), (0, 1)))
            @test f3 isa TabulatedFunction3D
        end
    end
end
