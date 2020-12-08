@testset "Models" begin
    @info "Testing models..."

    @testset "Model constructor errors" begin
        grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 1, 1))
        @test_throws TypeError IncompressibleModel(architecture=CPU, grid=grid)
        @test_throws TypeError IncompressibleModel(architecture=GPU, grid=grid)
        @test_throws TypeError IncompressibleModel(grid=grid, boundary_conditions=1)
        @test_throws TypeError IncompressibleModel(grid=grid, forcing=2)
        @test_throws TypeError IncompressibleModel(grid=grid, background_fields=3)
    end

    topos = ((Periodic, Periodic, Periodic),
             (Periodic, Periodic,  Bounded),
             (Periodic,  Bounded,  Bounded),
             (Bounded,   Bounded,  Bounded))

    for topo in topos
        @testset "$topo model construction" begin
            @info "  Testing $topo model construction..."
            for arch in archs, FT in float_types
		        arch isa GPU && topo == (Bounded, Bounded, Bounded) && continue

                grid = RegularCartesianGrid(FT, topology=topo, size=(16, 16, 2), extent=(1, 2, 3))
                model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT)

                # Just testing that the model was constructed with no errors/crashes.
                @test model isa IncompressibleModel

                # Test that the grid didn't get mangled
                @test grid === model.grid
            end
        end
    end

    @testset "Adjustment of halos in IncompressibleModel constructor" begin
        @info "  Testing adjustment of halos in IncompressibleModel constructor..."

        default_grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 2, 3))
        funny_grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 2, 3), halo=(1, 3, 4))

        # Model ensures that halos are at least of size 1
        model = IncompressibleModel(grid=default_grid)
        @test model.grid.Hx == 1 && model.grid.Hy == 1 && model.grid.Hz == 1

        model = IncompressibleModel(grid=funny_grid)
        @test model.grid.Hx == 1 && model.grid.Hy == 3 && model.grid.Hz == 4

        # Model ensures that halos are at least of size 2
        for scheme in (CenteredFourthOrder(), UpwindBiasedThirdOrder())
            model = IncompressibleModel(advection=scheme, grid=default_grid)
            @test model.grid.Hx == 2 && model.grid.Hy == 2 && model.grid.Hz == 2

            model = IncompressibleModel(advection=scheme, grid=funny_grid)
            @test model.grid.Hx == 2 && model.grid.Hy == 3 && model.grid.Hz == 4
        end

        # Model ensures that halos are at least of size 3
        for scheme in (WENO5(), UpwindBiasedFifthOrder())
            model = IncompressibleModel(advection=scheme, grid=default_grid)
            @test model.grid.Hx == 3 && model.grid.Hy == 3 && model.grid.Hz == 3

            model = IncompressibleModel(advection=scheme, grid=funny_grid)
            @test model.grid.Hx == 3 && model.grid.Hy == 3 && model.grid.Hz == 4
        end

        # Model ensures that halos are at least of size 2 with AnisotropicBiharmonicDiffusivity
        model = IncompressibleModel(closure=AnisotropicBiharmonicDiffusivity(), grid=default_grid)
        @test model.grid.Hx == 2 && model.grid.Hy == 2 && model.grid.Hz == 2

        model = IncompressibleModel(closure=AnisotropicBiharmonicDiffusivity(), grid=funny_grid)
        @test model.grid.Hx == 2 && model.grid.Hy == 3 && model.grid.Hz == 4
    end

    @testset "Model construction with single tracer and nothing tracer" begin
        @info "  Testing model construction with single tracer and nothing tracer..."
        for arch in archs
            grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 2, 3))

            model = IncompressibleModel(grid=grid, architecture=arch, tracers=:c, buoyancy=nothing)
            @test model isa IncompressibleModel

            model = IncompressibleModel(grid=grid, architecture=arch, tracers=nothing, buoyancy=nothing)
            @test model isa IncompressibleModel
        end
    end

    @testset "Non-dimensional model" begin
        @info "  Testing non-dimensional model construction..."
        for arch in archs, FT in float_types
            grid = RegularCartesianGrid(FT, size=(16, 16, 2), extent=(3, 2, 1))
            model = NonDimensionalModel(architecture=arch, float_type=FT, grid=grid, Re=1, Pr=1, Ro=Inf)

            # Just testing that a NonDimensionalModel was constructed with no errors/crashes.
            @test model isa IncompressibleModel
        end
    end

    @testset "Setting model fields" begin
        @info "  Testing setting model fields..."
        for arch in archs, FT in float_types
            N = (4, 4, 4)
            L = (2π, 3π, 5π)

            grid = RegularCartesianGrid(FT, size=N, extent=L)
            model = IncompressibleModel(architecture=arch, float_type=eltype(grid), grid=grid)

            u, v, w = model.velocities
            T, S = model.tracers

            # Test setting an array
            T₀ = rand(FT, size(grid)...)
            T_answer = deepcopy(T₀)

            set!(model; enforce_incompressibility=false, T=T₀)

            @test interior(T) ≈ T_answer

            # Test setting functions
            u₀(x, y, z) = 1 + x + y + z
            v₀(x, y, z) = 2 + sin(x * y * z)
            w₀(x, y, z) = 3 + y * z
            T₀(x, y, z) = 4 + tanh(x + y - z)
            S₀(x, y, z) = 5

            set!(model, enforce_incompressibility=false, u=u₀, v=v₀, w=w₀, T=T₀, S=S₀)

            xC, yC, zC = nodes((Cell, Cell, Cell), model.grid; reshape=true)
            xF, yF, zF = nodes((Face, Face, Face), model.grid; reshape=true)

            # Form solution arrays
            u_answer = u₀.(xF, yC, zC)
            v_answer = v₀.(xC, yF, zC)
            w_answer = w₀.(xC, yC, zF)
            T_answer = T₀.(xC, yC, zC)
            S_answer = S₀.(xC, yC, zC)

            Nx, Ny, Nz = size(model.grid)

            values_match = [
                            all(u_answer .≈ interior(model.velocities.u)),
                            all(v_answer .≈ interior(model.velocities.v)),
                            all(w_answer[:, :, 2:Nz] .≈ interior(model.velocities.w)[:, :, 2:Nz]),
                            all(T_answer .≈ interior(model.tracers.T)),
                            all(S_answer .≈ interior(model.tracers.S)),
                           ]

            @test all(values_match)

            # Test that update_state! works via u boundary conditions
            @test u[1, 1, 1] == u[Nx+1, 1, 1]  # x-periodicity
            @test u[1, 1, 1] == u[1, Ny+1, 1]  # y-periodicity
            @test all(u[1:Nx, 1:Ny, 1] .== u[1:Nx, 1:Ny, 0])     # free slip at bottom
            @test all(u[1:Nx, 1:Ny, Nz] .== u[1:Nx, 1:Ny, Nz+1]) # free slip at top

            # Test that enforce_incompressibility works
            set!(model, u=0, v=0, w=1, T=0, S=0)
            ϵ = 10 * eps(FT)
            @test all(abs.(interior(w)) .< ϵ)
        end
    end
end
