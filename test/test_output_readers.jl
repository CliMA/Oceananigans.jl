using Test
using JLD2

using Oceananigans
using Oceananigans.Units

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

    simulation.output_writers[:jld2_3d_without_halos] =
        JLD2OutputWriter(model, fields_to_output,
              prefix = "test_3d_output_without_halos",
        field_slicer = FieldSlicer(with_halos=false),
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

    filepath = "test_3d_output_with_halos.jld2"

    @testset "FieldTimeSeries{InMemory}" begin
        @info "  Testing FieldTimeSeries{InMemory}..."

        u = FieldTimeSeries(filepath, "u")
        v = FieldTimeSeries(filepath, "v")
        w = FieldTimeSeries(filepath, "w")
        T = FieldTimeSeries(filepath, "T")
        b = FieldTimeSeries(filepath, "b")
        ζ = FieldTimeSeries(filepath, "ζ")

        @test size(u) == (Nx, Ny, Nz,   Nt)
        @test size(v) == (Nx, Ny, Nz,   Nt)
        @test size(w) == (Nx, Ny, Nz+1, Nt)
        @test size(T) == (Nx, Ny, Nz,   Nt)
        @test size(b) == (Nx, Ny, Nz,   Nt)
        @test size(ζ) == (Nx, Ny, Nz,   Nt)

        @test u[1, 2, 3, 4] isa Number
        @test u[1] isa Field
        @test u[2] isa Field
    end

    @testset "FieldTimeSeries{OnDisk}" begin
        @info "  Testing FieldTimeSeries{OnDisk}..."

        ζ = FieldTimeSeries(filepath, "ζ", backend=OnDisk())
        @test ζ[1] isa Field
        @test ζ[2] isa Field
    end

    @testset "FieldTimeSeries{InMemory} reductions" begin
        @info "  Testing FieldTimeSeries{InMemory} reductions..."

        for name in ("u", "v", "w", "T", "b", "ζ"), fun in (sum, mean, maximum, minimum)
            f = FieldTimeSeries(filepath, name)

            ε = eps(maximum(f.data.parent))

            val1 = fun(f)
            val2 = fun([fun(f[n]) for n in 1:Nt])

            @test val1 ≈ val2 atol=ε
        end
    end

    for Backend in [InMemory, OnDisk]
        @testset "FieldDataset{$Backend}" begin
            @info "  Testing FieldDataset{$Backend}..."

            ds = FieldDataset(filepath, backend=Backend())

            @test ds isa Dict
            @test length(keys(ds)) == 8
            @test ds["u"] isa FieldTimeSeries
            @test ds["v"][1] isa Field
            @test ds["T"][2] isa Field
        end
    end
end
