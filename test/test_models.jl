function set_velocity_tracer_fields(arch, grid, fieldname, value, answer)
    model = IncompressibleModel(architecture=arch, float_type=eltype(grid), grid=grid)
    kwarg = Dict(fieldname=>value)
    set!(model; kwarg...)

    if fieldname ∈ propertynames(model.velocities)
        ϕ = getproperty(model.velocities, fieldname)
    else
        ϕ = getproperty(model.tracers, fieldname)
    end

    return interior(ϕ) ≈ answer
end

function initial_conditions_correctly_set(arch, FT)
    model = IncompressibleModel(grid=RegularCartesianGrid(FT, size=(16, 16, 8), extent=(1, 2, 3)),
                                architecture=arch, float_type=FT)

    # Set initial condition to some basic function we can easily check for.
    # We offset the functions by an integer so that we don't end up comparing
    # zero values with other zero values. I was too lazy to pick clever functions.
    u₀(x, y, z) = 1 + x + y + z
    v₀(x, y, z) = 2 + sin(x * y * z)
    w₀(x, y, z) = 3 + y * z
    T₀(x, y, z) = 4 + tanh(x + y - z)
    S₀(x, y, z) = 5

    set!(model, u=u₀, v=v₀, w=w₀, T=T₀, S=S₀)

    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    xC, yC, zC = nodes(model.tracers.T)
    xF, yF, zF = nodes((Face, Face, Face), model.grid)
    u, v, w = model.velocities.u.data, model.velocities.v.data, model.velocities.w.data
    T, S = model.tracers.T.data, model.tracers.S.data

    all_values_match = true
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        values_match = ( u[i, j, k] ≈ 1 + xF[i] + yC[j] + zC[k]       &&
                         v[i, j, k] ≈ 2 + sin(xC[i] * yF[j] * zC[k])  &&
                         w[i, j, k] ≈ 3 + yC[j] * zF[k]               &&
                         T[i, j, k] ≈ 4 + tanh(xC[i] + yC[j] - zC[k]) &&
                         S[i, j, k] ≈ 5)
        all_values_match = all_values_match & values_match
    end

    return all_values_match
end

@testset "Models" begin
    @info "Testing models..."

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

        # Model ensures that halos are at least of size 2 with CenteredFourthOrder advection
        for scheme in (CenteredFourthOrder(), UpwindBiasedThirdOrder())
            model = IncompressibleModel(advection=scheme, grid=default_grid)
            @test model.grid.Hx == 2 && model.grid.Hy == 2 && model.grid.Hz == 2

            model = IncompressibleModel(advection=scheme, grid=funny_grid)
            @test model.grid.Hx == 2 && model.grid.Hy == 3 && model.grid.Hz == 4
        end

        # Model ensures that halos are at least of size 3 with CenteredFourthOrder advection
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
            x, y, z = nodes((Face, Cell, Cell), grid, reshape=true)

            u₀(x, y, z) = x * y^2 * z^3
            u_answer = @. x * y^2 * z^3

            T₀ = rand(size(grid)...)
            T_answer = deepcopy(T₀)

            @test set_velocity_tracer_fields(arch, grid, :u, u₀, u_answer)
            @test set_velocity_tracer_fields(arch, grid, :T, T₀, T_answer)
            @test initial_conditions_correctly_set(arch, FT)
        end
    end
end
