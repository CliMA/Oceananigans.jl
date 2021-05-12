using Test
using JLD2

using Oceananigans
using Oceananigans.Units

using Oceananigans.Fields: location

function generate_some_interesting_simulation_data(Nx, Ny, Nz; architecture=CPU())
    grid = RegularRectilinearGrid(size=(Nx, Ny, Nz), extent=(64, 64, 32))

    Qʰ = 200
    ρₒ = 1026
    cᴾ = 3991
    Qᵀ = Qʰ / (ρₒ * cᴾ)
    dTdz = 0.01
    T_bcs = TracerBoundaryConditions(grid, top = FluxBoundaryCondition(Qᵀ), bottom = GradientBoundaryCondition(dTdz))

    u₁₀ = 10
    cᴰ = 2.5e-3
    ρₐ = 1.225
    Qᵘ = - ρₐ / ρₒ * cᴰ * u₁₀ * abs(u₁₀)
    u_bcs = UVelocityBoundaryConditions(grid, top = FluxBoundaryCondition(Qᵘ))

    @inline Qˢ(x, y, t, S, evaporation_rate) = - evaporation_rate * S
    evaporation_rate = 1e-3 / hour
    evaporation_bc = FluxBoundaryCondition(Qˢ, field_dependencies=:S, parameters=evaporation_rate)
    S_bcs = TracerBoundaryConditions(grid, top=evaporation_bc)

    model = IncompressibleModel(
               architecture = architecture,
                       grid = grid,
                   coriolis = FPlane(f=1e-4),
                   buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState()),
                    closure = IsotropicDiffusivity(ν=1e-2, κ=1e-2),
        boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs)
    )

    Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz)
    Tᵢ(x, y, z) = 20 + dTdz * z + dTdz * model.grid.Lz * 1e-6 * Ξ(z)
    uᵢ(x, y, z) = sqrt(abs(Qᵘ)) * 1e-3 * Ξ(z)
    set!(model, u=uᵢ, w=uᵢ, T=Tᵢ, S=35)

    wizard = TimeStepWizard(cfl=1.0, Δt=10.0, max_change=1.1, max_Δt=1minute)
    simulation = Simulation(model, Δt=wizard, stop_time=2minutes, iteration_interval=1,
                            progress= sim -> @info "Time $(sim.model.clock.time)")

    # LOTS OF OUTPUT

    u, v, w = model.velocities

    computed_fields = (
        b = BuoyancyField(model),
        ζ = ComputedField(∂x(v) - ∂y(u)),
        ke = ComputedField(√(u^2 + v^2))
    )

    fields_to_output = merge(model.velocities, model.tracers, computed_fields)

    simulation.output_writers[:jld2_3d_with_halos] =
        JLD2OutputWriter(model, fields_to_output,
                  prefix = "test_3d_output_with_halos",
            field_slicer = FieldSlicer(with_halos=true),
                schedule = TimeInterval(30seconds),
                   force = true)

    profiles = NamedTuple{keys(fields_to_output)}(AveragedField(f, dims=(1, 2)) for f in fields_to_output)

    simulation.output_writers[:jld2_1d_with_halos] =
        JLD2OutputWriter(model, profiles,
                  prefix = "test_1d_output_with_halos",
            field_slicer = FieldSlicer(with_halos=true),
                schedule = TimeInterval(30seconds),
                   force = true)

    run!(simulation)

    return nothing
end

@testset "OutputReaders" begin
    @info "Testing output readers..."

    Nx, Ny, Nz = 16, 10, 5
    generate_some_interesting_simulation_data(Nx, Ny, Nz)
    Nt = 5

    filepath3d = "test_3d_output_with_halos.jld2"
    filepath1d = "test_1d_output_with_halos.jld2"

    @testset "FieldTimeSeries{InMemory}" begin
        @info "  Testing FieldTimeSeries{InMemory}..."

        ## 3D Fields

        u3 = FieldTimeSeries(filepath3d, "u")
        v3 = FieldTimeSeries(filepath3d, "v")
        w3 = FieldTimeSeries(filepath3d, "w")
        T3 = FieldTimeSeries(filepath3d, "T")
        b3 = FieldTimeSeries(filepath3d, "b")
        ζ3 = FieldTimeSeries(filepath3d, "ζ")

        @test location(u3) == (Face, Center, Center)
        @test location(v3) == (Center, Face, Center)
        @test location(w3) == (Center, Center, Face)
        @test location(T3) == (Center, Center, Center)
        @test location(b3) == (Center, Center, Center)
        @test location(ζ3) == (Face, Face, Center)

        @test size(u3) == (Nx, Ny, Nz,   Nt)
        @test size(v3) == (Nx, Ny, Nz,   Nt)
        @test size(w3) == (Nx, Ny, Nz+1, Nt)
        @test size(T3) == (Nx, Ny, Nz,   Nt)
        @test size(b3) == (Nx, Ny, Nz,   Nt)
        @test size(ζ3) == (Nx, Ny, Nz,   Nt)

        @test u3[1, 2, 3, 4] isa Number
        @test u3[1] isa Field
        @test v3[2] isa Field

        ## 1D AveragedFields

        u1 = FieldTimeSeries(filepath1d, "u")
        v1 = FieldTimeSeries(filepath1d, "v")
        w1 = FieldTimeSeries(filepath1d, "w")
        T1 = FieldTimeSeries(filepath1d, "T")
        b1 = FieldTimeSeries(filepath1d, "b")
        ζ1 = FieldTimeSeries(filepath1d, "ζ")

        @test location(u1) == (Nothing, Nothing, Center)
        @test location(v1) == (Nothing, Nothing, Center)
        @test location(w1) == (Nothing, Nothing, Face)
        @test location(T1) == (Nothing, Nothing, Center)
        @test location(b1) == (Nothing, Nothing, Center)
        @test location(ζ1) == (Nothing, Nothing, Center)

        @test size(u1) == (1, 1, Nz,   Nt)
        @test size(v1) == (1, 1, Nz,   Nt)
        @test size(w1) == (1, 1, Nz+1, Nt)
        @test size(T1) == (1, 1, Nz,   Nt)
        @test size(b1) == (1, 1, Nz,   Nt)
        @test size(ζ1) == (1, 1, Nz,   Nt)

        @test u1[1, 1, 3, 4] isa Number
        @test u1[1] isa Field
        @test v1[2] isa Field
    end

    @testset "FieldTimeSeries{OnDisk}" begin
        @info "  Testing FieldTimeSeries{OnDisk}..."

        ζ = FieldTimeSeries(filepath3d, "ζ", backend=OnDisk())
        @test location(ζ) == (Face, Face, Center)
        @test size(ζ) == (Nx, Ny, Nz, Nt)
        @test ζ[1] isa Field
        @test ζ[2] isa Field

        b = FieldTimeSeries(filepath1d, "b", backend=OnDisk())
        @test location(b) == (Nothing, Nothing, Center)
        @test size(b) == (1, 1, Nz, Nt)
        @test b[1] isa Field
        @test b[2] isa Field
    end

    @testset "FieldTimeSeries{InMemory} reductions" begin
        @info "  Testing FieldTimeSeries{InMemory} reductions..."

        for name in ("u", "v", "w", "T", "b", "ζ"), fun in (sum, mean, maximum, minimum)
            f = FieldTimeSeries(filepath3d, name)

            ε = eps(maximum(f.data.parent))

            val1 = fun(f)
            val2 = fun([fun(f[n]) for n in 1:Nt])

            @test val1 ≈ val2 atol=ε
        end
    end

    for Backend in [InMemory, OnDisk]
        @testset "FieldDataset{$Backend}" begin
            @info "  Testing FieldDataset{$Backend}..."

            ds = FieldDataset(filepath3d, backend=Backend())

            @test ds isa Dict
            @test length(keys(ds)) == 8
            @test ds["u"] isa FieldTimeSeries
            @test ds["v"][1] isa Field
            @test ds["T"][2] isa Field
        end
    end

    rm("test_3d_output_with_halos.jld2")
    rm("test_1d_output_with_halos.jld2")
end
