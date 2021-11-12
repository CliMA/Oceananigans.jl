using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization

@testset "Immersed boundaries with hydrostatic free surface models" begin
    @info "Testing immersed boundaries vertical integrals"

    for arch in archs
        Nx = 5
        Ny = 5

        # A spherical domain
        underlying_grid =
        RegularRectilinearGrid(size=(Nx, Ny, 3), extent=(Nx, Ny, 3), topology=(Periodic,Periodic,Bounded))

        B = [-3. for i=1:Nx, j=1:Ny ]
        B[2:Nx-1,2:Ny-1] .= [-2. for i=2:Nx-1, j=2:Ny-1 ]
        B[3:Nx-2,3:Ny-2] .= [-1. for i=3:Nx-2, j=3:Ny-2 ]
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(B))

        free_surface = ImplicitFreeSurface(gravitational_acceleration=0.1)

        model = HydrostaticFreeSurfaceModel(grid = grid,
                                           architecture = arch,
                                           #free_surface = ExplicitFreeSurface(),
                                           #free_surface = ImplicitFreeSurface(maximum_iterations=10),
                                           free_surface = ImplicitFreeSurface(),
                                           momentum_advection = nothing,
                                           tracer_advection = WENO5(),
                                           coriolis = nothing,
                                           buoyancy = nothing,
                                           tracers = nothing,
                                           closure = nothing)

        x_ref = [0.0  0.0  0.0  0.0  0.0  0.0  0.0
                 0.0  3.0  3.0  3.0  3.0  3.0  0.0
                 0.0  3.0  2.0  2.0  2.0  2.0  0.0
                 0.0  3.0  2.0  1.0  1.0  2.0  0.0
                 0.0  3.0  2.0  2.0  2.0  2.0  0.0
                 0.0  3.0  3.0  3.0  3.0  3.0  0.0
                 0.0  0.0  0.0  0.0  0.0  0.0  0.0]'

        y_ref = [0.0  0.0  0.0  0.0  0.0  0.0  0.0
                 0.0  3.0  3.0  3.0  3.0  3.0  0.0
                 0.0  3.0  2.0  2.0  2.0  3.0  0.0
                 0.0  3.0  2.0  1.0  2.0  3.0  0.0
                 0.0  3.0  2.0  1.0  2.0  3.0  0.0
                 0.0  3.0  2.0  2.0  2.0  3.0  0.0
                 0.0  0.0  0.0  0.0  0.0  0.0  0.0]'

        fs=model.free_surface
        xok=parent(fs.implicit_step_solver.vertically_integrated_lateral_areas.xᶠᶜᶜ.data[:,:,1])-x_ref == zeros(7,7)
        yok=parent(fs.implicit_step_solver.vertically_integrated_lateral_areas.yᶜᶠᶜ.data[:,:,1])-y_ref == zeros(7,7)
        @test (xok & yok)
    end
end

