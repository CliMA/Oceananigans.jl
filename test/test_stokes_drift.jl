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

    @testset "StokesDrift(; uˢ, vˢ)" begin
        for arch in archs
            grid = RectilinearGrid(arch; size=(4, 4, 8), halo=(3, 3, 3),
                                   x=(0, 4), y=(0, 4), z=(-1, 0),
                                   topology=(Periodic, Periodic, Bounded))

            uˢ = Field{Face,   Center, Center}(grid)
            vˢ = Field{Center, Face,   Center}(grid)
            sd = StokesDrift(; uˢ, vˢ)

            @test sd.uˢ === uˢ
            @test sd.vˢ === vˢ
            @test Oceananigans.StokesDrifts.∂t_uˢ(2, 2, 2, grid, sd, 0.0) == 0
            @test Oceananigans.StokesDrifts.∂t_vˢ(2, 2, 2, grid, sd, 0.0) == 0
            @test Oceananigans.StokesDrifts.∂t_wˢ(2, 2, 2, grid, sd, 0.0) == 0

            fake_U = (u=Field{Face,   Center, Center}(grid),
                      v=Field{Center, Face,   Center}(grid),
                      w=Field{Center, Center, Face  }(grid))
            @test Oceananigans.StokesDrifts.x_curl_Uˢ_cross_U(2, 2, 2, grid, sd, fake_U, 0.0) == 0
            @test Oceananigans.StokesDrifts.y_curl_Uˢ_cross_U(2, 2, 2, grid, sd, fake_U, 0.0) == 0
            @test Oceananigans.StokesDrifts.z_curl_Uˢ_cross_U(2, 2, 2, grid, sd, fake_U, 0.0) == 0

            # Linear vertical shear uˢ = α·z and uniform w = W gives
            # x_curl_Uˢ_cross_U = W·α at every interior point.
            α, W = 0.7, 0.3
            set!(uˢ, (x, y, z) -> α * z); Oceananigans.BoundaryConditions.fill_halo_regions!(uˢ)
            set!(fake_U.w, W); Oceananigans.BoundaryConditions.fill_halo_regions!(fake_U.w)
            for k in 2:grid.Nz - 1, j in 2:grid.Ny - 1, i in 2:grid.Nx - 1
                @test Oceananigans.StokesDrifts.x_curl_Uˢ_cross_U(i, j, k, grid, sd, fake_U, 0.0) ≈ W * α atol=1e-12
            end

            # Cross-derivative term: uˢ = β·y, v = V, w = 0 →
            # x_curl_Uˢ_cross_U = -V·(∂x vˢ - ∂y uˢ) = V·β.
            # UniformStokesDrift would drop this term entirely.
            β, V = 0.5, 0.2
            set!(uˢ, (x, y, z) -> β * y); Oceananigans.BoundaryConditions.fill_halo_regions!(uˢ)
            set!(fake_U.w, 0); Oceananigans.BoundaryConditions.fill_halo_regions!(fake_U.w)
            set!(fake_U.v, V); Oceananigans.BoundaryConditions.fill_halo_regions!(fake_U.v)
            for k in 2:grid.Nz - 1, j in 2:grid.Ny - 1, i in 2:grid.Nx - 1
                @test Oceananigans.StokesDrifts.x_curl_Uˢ_cross_U(i, j, k, grid, sd, fake_U, 0.0) ≈ V * β atol=1e-12
            end
        end
    end

    @testset "StokesDrift(; ∂t_uˢ, ∂t_vˢ) as Fields" begin
        for arch in archs
            grid = RectilinearGrid(arch; size=(4, 4, 8), halo=(3, 3, 3),
                                   x=(0, 4), y=(0, 4), z=(-1, 0),
                                   topology=(Periodic, Periodic, Bounded))

            ∂t_uˢ = Field{Face,   Center, Center}(grid)
            ∂t_vˢ = Field{Center, Face,   Center}(grid)
            sd = StokesDrift(; ∂t_uˢ, ∂t_vˢ)

            @test sd.∂t_uˢ === ∂t_uˢ
            @test sd.∂t_vˢ === ∂t_vˢ

            # Zero fields → zero Stokes acceleration.
            @test Oceananigans.StokesDrifts.∂t_uˢ(2, 2, 2, grid, sd, 0.0) == 0
            @test Oceananigans.StokesDrifts.∂t_vˢ(2, 2, 2, grid, sd, 0.0) == 0

            # Constant Field → that constant comes through at every interior point.
            a_u, a_v = 0.13, -0.27
            set!(∂t_uˢ, a_u); Oceananigans.BoundaryConditions.fill_halo_regions!(∂t_uˢ)
            set!(∂t_vˢ, a_v); Oceananigans.BoundaryConditions.fill_halo_regions!(∂t_vˢ)
            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                @test Oceananigans.StokesDrifts.∂t_uˢ(i, j, k, grid, sd, 0.0) ≈ a_u atol=1e-14
                @test Oceananigans.StokesDrifts.∂t_vˢ(i, j, k, grid, sd, 0.0) ≈ a_v atol=1e-14
            end

            # Field paths for ∂t and uˢ/vˢ compose without ambiguity.
            uˢ = Field{Face,   Center, Center}(grid)
            vˢ = Field{Center, Face,   Center}(grid)
            sd2 = StokesDrift(; uˢ, vˢ, ∂t_uˢ, ∂t_vˢ)
            @test sd2.uˢ === uˢ
            @test sd2.∂t_uˢ === ∂t_uˢ
            @test Oceananigans.StokesDrifts.∂t_uˢ(2, 2, 2, grid, sd2, 0.0) ≈ a_u atol=1e-14
            @test Oceananigans.StokesDrifts.∂t_vˢ(2, 2, 2, grid, sd2, 0.0) ≈ a_v atol=1e-14
        end
    end
end
