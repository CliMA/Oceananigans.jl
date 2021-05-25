using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization

@testset "Immersed boundaries with hydrostatic free surface models" begin
    @info "Testing immersed boundaries with hydrostatic free surface models..."

    for arch in archs
        underlying_grid = RegularRectilinearGrid(size=(8, 8, 8), x = (-5, 5), y = (-5, 5), z = (0, 2))

        bump(x, y, z) = z < exp(-x^2 - y^2)

        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(bump))
        
        for closure in (IsotropicDiffusivity(ν=1, κ=0.5),
                        IsotropicDiffusivity(ν=1, κ=0.5, time_discretization=VerticallyImplicitTimeDiscretization()))

            model = HydrostaticFreeSurfaceModel(grid = grid, 
                                                architecture = arch, 
                                                tracers = :b,
                                                buoyancy = BuoyancyTracer(),
                                                closure = closure)

            u = model.velocities.u
            b = model.tracers.b

            # Linear stratification
            set!(model, u = 1, b = (x, y, z) -> 4 * z)
        
            # Inside the bump
            @test b[4, 4, 2] == 0 
            @test u[4, 4, 2] == 0

            simulation = Simulation(model, Δt = 1e-3, stop_iteration=2)

            run!(simulation)

            # Inside the bump
            @test b[4, 4, 2] == 0
            @test u[4, 4, 2] == 0
        end
    end
end

