include("dependencies_for_runtests.jl")

∂t_uˢ_uniform(z, t, h) = exp(z / h) * cos(t)
∂t_vˢ_uniform(z, t, h) = exp(z / h) * cos(t)
∂z_uˢ_uniform(z, t, h) = exp(z / h) / h * sin(t)
∂z_vˢ_uniform(z, t, h) = exp(z / h) / h * sin(t)

∂t_uˢ(x, y, z, t, h) = exp(z / h) * cos(t)
∂t_vˢ(x, y, z, t, h) = exp(z / h) * cos(t)
∂t_wˢ(x, y, z, t, h) = 0
∂x_vˢ(x, y, z, t, h) = 0
∂x_wˢ(x, y, z, t, h) = 0
∂y_uˢ(x, y, z, t, h) = 0
∂y_wˢ(x, y, z, t, h) = 0
∂z_uˢ(x, y, z, t, h) = exp(z / h) / h * sin(t)
∂z_vˢ(x, y, z, t, h) = exp(z / h) / h * sin(t)

function instantiate_uniform_stokes_drift()
    stokes_drift = UniformStokesDrift(∂t_uˢ = ∂t_uˢ_uniform,
                                      ∂t_vˢ = ∂t_vˢ_uniform,
                                      ∂z_uˢ = ∂z_uˢ_uniform,
                                      ∂z_vˢ = ∂z_vˢ_uniform,
                                      parameters = 20)

    return true
end

function instantiate_stokes_drift()
    stokes_drift = StokesDrift(∂t_uˢ = ∂t_uˢ,
                               ∂t_vˢ = ∂t_vˢ,
                               ∂t_wˢ = ∂t_wˢ,
                               ∂x_vˢ = ∂x_vˢ,
                               ∂x_wˢ = ∂x_wˢ,
                               ∂y_uˢ = ∂y_uˢ,
                               ∂y_wˢ = ∂y_wˢ,
                               ∂z_uˢ = ∂z_uˢ,
                               ∂z_vˢ = ∂z_vˢ,
                               parameters = 20)

    return true
end

@testset "Stokes drift" begin
    @info "Testing Stokes drift..."

    @testset "Stokes drift" begin
        @test instantiate_uniform_stokes_drift()
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

