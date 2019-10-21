function instantiate_fplane(T)
    coriolis = FPlane(T, f=π)
    return coriolis.f == T(π)
end

function instantiate_betaplane_1(T)
    coriolis = BetaPlane(T, f₀=π, β=2π)
    return coriolis.f₀ == T(π)
end

function instantiate_betaplane_2(T)
    coriolis = BetaPlane(T, latitude=70, radius=2π, rotation_rate=3π)
    return coriolis.f₀ == T(6π * sind(70))
end


@testset "Coriolis" begin
    println("Testing Coriolis...")

    @testset "Coriolis" begin
        for T in float_types
            @test instantiate_fplane(T)
            @test instantiate_betaplane_1(T)
            @test instantiate_betaplane_2(T)
        end
    end
end
