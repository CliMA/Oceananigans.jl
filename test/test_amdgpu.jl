include("dependencies_for_runtests.jl")

using AMDGPU
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

@testset "AMDGPU with HydrostaticFreeSurfaceModel" begin
    roc = AMDGPU.ROCBackend()
    arch = GPU(roc)

    for FT in float_types
        @info "    Testing on $arch with $FT"

        grid = LatitudeLongitudeGrid(arch, FT, size=(4, 8, 16), longitude=(-60, 60), latitude=(0, 60), z=(0, 1))

        @test parent(grid.λᶠᵃᵃ) isa ROCArray
        @test parent(grid.λᶜᵃᵃ) isa ROCArray
        @test parent(grid.φᵃᶠᵃ) isa ROCArray
        @test parent(grid.φᵃᶜᵃ) isa ROCArray
        @test parent(grid.zᵃᵃᶠ) isa ROCArray
        @test parent(grid.zᵃᵃᶜ) isa ROCArray
        @test eltype(grid) == FT
        @test architecture(grid) isa GPU

        equation_of_state = TEOS10EquationOfState()
        buoyancy = SeawaterBuoyancy(; equation_of_state)

        model = HydrostaticFreeSurfaceModel(; grid, buoyancy,
                                            coriolis = FPlane(latitude=45),
                                            tracers = (:T, :S),
                                            momentum_advection = WENO(order=5),
                                            tracer_advection = WENO(order=5),
                                            free_surface = SplitExplicitFreeSurface(grid; substeps=60))

        for field in merge(model.velocities, model.tracers)
            @test parent(field) isa ROCArray
        end

        simulation = Simulation(model, Δt=1minute, stop_iteration=3)
        run!(simulation)

        @test iteration(simulation) == 3
        @test time(simulation) == 3minutes
    end
end

@testset "AMDGPU with NonhydrostaticModel" begin
    roc = AMDGPU.ROCBackend()
    arch = GPU(roc)

    for FT in float_types
        @info "    Testing on $arch with $FT"

        z = 0:16 |> collect
        grid = RectilinearGrid(arch, FT, size=(4, 8, 16), x=(0, 1), y=(0, 1), z=z)

        @test parent(grid.xᶠᵃᵃ) isa ROCArray
        @test parent(grid.xᶜᵃᵃ) isa ROCArray
        @test eltype(grid) == FT
        @test architecture(grid) isa GPU

        model = NonhydrostaticModel(; grid,
                                    coriolis = FPlane(latitude=45),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    advection = WENO(order=5))

        for field in merge(model.velocities, model.tracers)
            @test parent(field) isa ROCArray
        end

        simulation = Simulation(model, Δt=1minute, stop_iteration=3)
        run!(simulation)

        @test iteration(simulation) == 3
        @test time(simulation) == 3minutes
    end
end
