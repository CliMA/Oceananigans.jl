@testset "Models" begin
    println("Testing models...")

    @testset "Doubly periodic model" begin
        println("  Testing doubly periodic model construction...")
        for arch in archs, FT in float_types
            model = Model(N=(4, 5, 6), L=(1, 2, 3), arch=arch, float_type=FT)

            # Just testing that a Model was constructed with no errors/crashes.
            @test true
        end
    end

    @testset "Reentrant channel model" begin
        println("  Testing reentrant channel model construction...")
        for arch in archs, FT in float_types
            model = ChannelModel(N=(6, 5, 4), L=(3, 2, 1), arch=arch, float_type=FT)

            # Just testing that a ChannelModel was constructed with no errors/crashes.
            @test true
        end
    end
end
