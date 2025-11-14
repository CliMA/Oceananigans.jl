include("dependencies_for_runtests.jl")

using AMDGPU

function build_and_timestep_simulation(model)
    FT = eltype(model)

    for field in merge(model.velocities, model.tracers)
        @test parent(field) isa ROCArray
    end

    simulation = Simulation(model, Δt=1minute, stop_iteration=3, verbose=false)
    run!(simulation)

    @test iteration(simulation) == 3
    @test time(simulation) ≈ FT(3minutes)

    return nothing
end

@testset "AMDGPU extension" begin
    roc = AMDGPU.ROCBackend()
    arch = GPU(roc)

    for FT in float_types
        @info "Testing grids on $arch with $FT..."

        regular_grid = RectilinearGrid(arch, FT, size=(4, 8, 16), x=(0, 4), y=(0, 1), z=(0, 16))
        horizontally_stretched_grid = RectilinearGrid(arch, FT, size=(4, 8, 16), x=[0, 1, 2, 3, 4], y=(0, 1), z=(0, 16))
        vertically_stretched_grid = RectilinearGrid(arch, FT, size=(16, 8, 4), x=(0, 16), y=(0, 1), z=[0, 1, 2, 3, 4])

        @test parent(horizontally_stretched_grid.xᶠᵃᵃ) isa ROCArray
        @test parent(horizontally_stretched_grid.xᶜᵃᵃ) isa ROCArray

        @test parent(vertically_stretched_grid.z.cᵃᵃᶠ) isa ROCArray
        @test parent(vertically_stretched_grid.z.cᵃᵃᶜ) isa ROCArray
        @test parent(vertically_stretched_grid.z.Δᵃᵃᶠ) isa ROCArray
        @test parent(vertically_stretched_grid.z.Δᵃᵃᶜ) isa ROCArray

        for grid in (regular_grid, horizontally_stretched_grid, vertically_stretched_grid)
            @test eltype(grid) == FT
            @test architecture(grid) isa GPU
        end

        @info "Testing HydrostaticFreeSurfaceModel on $arch with $FT..."

        coriolis = FPlane(latitude=45)
        buoyancy = BuoyancyTracer()
        tracers = :b
        advection = WENO(order=5)

        for grid in (regular_grid, horizontally_stretched_grid, vertically_stretched_grid)
            momentum_advection = tracer_advection = advection

            free_surface = SplitExplicitFreeSurface(grid; substeps=60)

            model = HydrostaticFreeSurfaceModel(; grid, free_surface,
                                                coriolis, buoyancy, tracers,
                                                momentum_advection, tracer_advection)

            build_and_timestep_simulation(model)
        end

        @info "Testing NonhydrostaticModel on $arch with $FT..."

        for grid in (regular_grid, vertically_stretched_grid)
            pressure_solvers = (Oceananigans.Solvers.ConjugateGradientPoissonSolver(grid, maxiter=10; reltol=1e-7, abstol=1e-7, preconditioner=nothing),
                                Oceananigans.Solvers.FFTBasedPoissonSolver(grid))

            for pressure_solver in pressure_solvers
                model = NonhydrostaticModel(; grid, pressure_solver,
                                            coriolis, buoyancy,
                                            tracers, advection)

                build_and_timestep_simulation(model)
            end
        end
    end
end
