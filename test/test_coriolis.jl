function instantiate_coriolis(T)
    coriolis = FPlane(T, f=π)
    return coriolis.f == T(π)
end

@testset "Coriolis" begin
    println("Testing Coriolis...")

    @testset "Coriolis" begin
        for T in float_types
            @test instantiate_coriolis(T)
        end
    end
end
