function instantiate_stokes_drift()
    ∂t_uˢ(z, t) = exp(z/20) * cos(t)
    ∂t_vˢ(z, t) = exp(z/20) * cos(t)
    ∂z_uˢ(z, t) = exp(z/20) * cos(t)
    ∂z_vˢ(z, t) = exp(z/20) * cos(t)
    stokes_drift = UniformStokesDrift(∂t_uˢ=∂t_uˢ, ∂t_vˢ=∂t_vˢ, ∂z_uˢ=∂z_uˢ, ∂z_vˢ=∂z_vˢ)
    return true
end

@testset "Stokes drift" begin
    @info "Testing Stokes drift..."

    @testset "Stokes drift" begin
        @test instantiate_stokes_drift()

        for arch in archs
            grid = RectilinearGrid(arch, size=(3, 3, 3), extent=(1, 1, 1))
            stokes_drift = UniformStokesDrift(grid)
            @test location(stokes_drift.∂z_uˢ) === (Nothing, Nothing, Face)
            @test location(stokes_drift.∂z_vˢ) === (Nothing, Nothing, Face)
            @test location(stokes_drift.∂t_uˢ) === (Nothing, Nothing, Center)
            @test location(stokes_drift.∂t_vˢ) === (Nothing, Nothing, Center)
        end
    end
end

