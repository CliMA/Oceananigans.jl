function instantiate_surface_waves()
    ∂t_uˢ(z, t) = exp(z/20) * cos(t)
    ∂t_vˢ(z, t) = exp(z/20) * cos(t)
    ∂z_uˢ(z, t) = exp(z/20) * cos(t)
    ∂z_vˢ(z, t) = exp(z/20) * cos(t)
    surface_waves = SurfaceWaves.UniformStokesDrift(∂t_uˢ=∂t_uˢ, ∂t_vˢ=∂t_vˢ,
                                                    ∂z_uˢ=∂z_uˢ, ∂z_vˢ=∂z_vˢ)
                                                    
    return true
end

@testset "Surface waves" begin
    println("Testing surface waves...")

    @testset "Surface waves" begin
        @test instantiate_surface_waves()
    end
end
