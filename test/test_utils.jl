include("dependencies_for_runtests.jl")

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
        @test prettytime(year) == "1 year"

        @test prettytime(2second) == "2 seconds"
        @test prettytime(4minute) == "4 minutes"
        @test prettytime(8hour) == "8 hours"
        @test prettytime(16day) == "16 days"
        @test prettytime(32year) == "32 years"

        @test prettytime(13.7seconds) == "13.700 seconds"
        @test prettytime(6.666minutes) == "6.666 minutes"
        @test prettytime(1.234hour) == "1.234 hours"
        @test prettytime(40.5days) == "40.500 days"
        @test prettytime(5.0001years) == "5.000 years"
    end
end
