include("dependencies_for_runtests.jl")

using AMDGPU

@testset "AMDGPU extension" begin
    roc = AMDGPU.ROCBackend()
    arch = GPU(roc)

    for FT in float_types
        @info "    Testing HydrostaticFreeSurfaceModel on $arch with $FT"

        grid = RectilinearGrid(arch, FT, size=(4, 8, 16), x=[0, 1, 2, 3, 4], y=(0, 1), z=(0, 16))

        @test parent(grid.xᶠᵃᵃ) isa ROCArray
        @test parent(grid.xᶜᵃᵃ) isa ROCArray
        @test eltype(grid) == FT
        @test architecture(grid) isa GPU

        model = HydrostaticFreeSurfaceModel(; grid,
                                            coriolis = FPlane(latitude=45),
                                            buoyancy = BuoyancyTracer(),
                                            tracers = :b,
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

        @info "    Testing NonhydrostaticModel on $arch with $FT"

        pressure_solvers = (Oceananigans.Solvers.ConjugateGradientPoissonSolver(grid, maxiter=10; reltol=1e-7, abstol=1e-7, preconditioner=nothing),
                            Oceananigans.Solvers.FFTBasedPoissonSolver(grid))

        for pressure_solver in pressure_solvers

            model = NonhydrostaticModel(; grid, pressure_solver
                                        coriolis = FPlane(latitude=45),
                                        buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                                        momentum_advection = WENO(order=5),
                                        tracer_advection = WENO(order=5),)

            for field in merge(model.velocities, model.tracers)
                @test parent(field) isa ROCArray
            end

            simulation = Simulation(model, Δt=1minute, stop_iteration=3)
            run!(simulation)

            @test iteration(simulation) == 3
            @test time(simulation) == 3minutes
        end
    end
end
