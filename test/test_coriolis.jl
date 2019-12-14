function instantiate_fplane_1(FT)
    coriolis = FPlane(FT, f=π)
    return coriolis.f == FT(π)
end

function instantiate_fplane_2(FT)
    coriolis = FPlane(FT, rotation_rate=2, latitude=30)
    return coriolis.f == FT(2)
end

function instantiate_betaplane_1(FT)
    coriolis = BetaPlane(FT, f₀=π, β=2π)
    return coriolis.f₀ == FT(π)
end

function instantiate_betaplane_2(FT)
    coriolis = BetaPlane(FT, latitude=70, radius=2π, rotation_rate=3π)
    return coriolis.f₀ == FT(6π * sind(70))
end

@testset "Coriolis" begin
    println("Testing Coriolis...")

    @testset "Coriolis" begin
        for FT in float_types
            @test instantiate_fplane_1(FT)
            @test instantiate_fplane_2(FT)
            @test instantiate_betaplane_1(FT)
            @test instantiate_betaplane_2(FT)

            # Test that FPlane throws an ArgumentError
            @test_throws ArgumentError FPlane(FT, rotation_rate=7e-5)
            @test_throws ArgumentError FPlane(FT, f=1, latitude=40)
            @test_throws ArgumentError FPlane(FT, f=1, rotation_rate=7e-5, latitude=40)

            # Non-exhaustively test that BetaPlane throws an ArgumentError
            @test_throws ArgumentError BetaPlane(FT, f₀=1)
            @test_throws ArgumentError BetaPlane(FT, β=1)
            @test_throws ArgumentError BetaPlane(FT, f₀=1e-4, β=1e-11, latitude=70)
        end
    end
end
