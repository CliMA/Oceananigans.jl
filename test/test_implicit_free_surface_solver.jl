using Statistics
using Oceananigans.BuoyancyModels: g_Earth
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface, FreeSurface,
      compute_volume_scaled_divergence!, add_previous_free_surface_contribution,
      compute_vertically_integrated_volume_flux!

function run_implicit_free_surface_solver_tests(arch, grid)
    ### Nx = 360
    ### Ny = 360
    # A spherical domain with bounded toplogy
    ### grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
    ###                                 longitude = (-30, 30),
    ###                                 latitude = (15, 75),
    ###                                 z = (-4000, 0))

    # A boring grid
    ## RegularRectilinearGrid(FT, size=(3, 1, 4), extent=(3, 1, 4))
    ### grid = RegularRectilinearGrid(size = (128, 1, 1),
    ###                         x = (0, 1000kilometers), y = (0, 1), z = (-400, 0),
    ###                         topology = (Bounded, Periodic, Bounded))
    Nx = grid.Nx
    Ny = grid.Ny
    Δt = 900

    free_surface = ImplicitFreeSurface(gravitational_acceleration=g_Earth)

    # Create fields with default bounded and zeroflux boundaries.
    velocities = VelocityFields(arch, grid)
    η          = CenterField(arch, grid, TracerBoundaryConditions(grid))
    @. η.data  = 0

    # Initialize the solver
    free_surface = FreeSurface(free_surface, velocities, arch, grid)

    # Create a divergent velocity
    u, v, w = velocities
    @. u.data = 0
    @. v.data = 0
    @. w.data = 0
    imid = Int(floor(grid.Nx / 2)) + 1
    jmid = Int(floor(grid.Ny / 2)) + 1
    CUDA.@allowscalar u.data[imid, jmid, 1] = 1

    fill_halo_regions!(u.data, u.boundary_conditions, arch, grid, nothing, nothing)
    fill_halo_regions!(v.data, v.boundary_conditions, arch, grid, nothing, nothing)
    fill_halo_regions!(w.data, w.boundary_conditions, arch, grid, nothing, nothing)

    # Create a fake model
    model = (architecture=arch,grid=grid,free_surface=free_surface, Δt=Δt, velocities=velocities)

    ## We need vertically integrated U,V
    event = compute_vertically_integrated_volume_flux!(free_surface, model)
    wait(device(model.architecture), event)
    u=free_surface.barotropic_volume_flux.u
    v=free_surface.barotropic_volume_flux.v
    fill_halo_regions!(u.data ,u.boundary_conditions, arch, grid, nothing, nothing )
    fill_halo_regions!(v.data ,v.boundary_conditions, arch, grid, nothing, nothing )

    ### We don't need the halo below, its just here for some debugging
    Ax=free_surface.vertically_integrated_lateral_face_areas.Ax
    Ay=free_surface.vertically_integrated_lateral_face_areas.Ay
    fill_halo_regions!(Ax.data ,Ax.boundary_conditions, arch, grid, nothing, nothing )
    fill_halo_regions!(Ay.data ,Ay.boundary_conditions, arch, grid, nothing, nothing )

    # Calculate the vertically ingrated divergence term
    event = compute_volume_scaled_divergence!(free_surface, model)
    wait(device(model.architecture), event)

    # Scale and add in η term
    RHS = free_surface.implicit_step_solver.solver.settings.RHS
    RHS .= RHS/(free_surface.gravitational_acceleration*Δt)
    event = add_previous_free_surface_contribution(free_surface, model, Δt )
    wait(device(model.architecture), event)
    fill_halo_regions!(RHS   , η.boundary_conditions, model.architecture, model.grid)

    x  = free_surface.implicit_step_solver.solver.settings.x
    x .= η.data
    fill_halo_regions!(x ,η.boundary_conditions, model.architecture, model.grid)
    solve_poisson_equation!(free_surface.implicit_step_solver.solver, RHS, x; Δt=Δt, g=free_surface.gravitational_acceleration)
    ## exit()
    fill_halo_regions!(x ,η.boundary_conditions, model.architecture, model.grid)
    free_surface.η.data .= x

    # Amatrix_function!(result, x, arch, grid, bcs; args...)
    result = free_surface.implicit_step_solver.solver.settings.A(x;Δt=Δt,g=free_surface.gravitational_acceleration)

    CUDA.@allowscalar begin
     @test abs(minimum(result[1:Nx, 1:Ny, 1] .- RHS[1:Nx, 1:Ny, 1])) < 1e-11
     @test abs(maximum(result[1:Nx, 1:Ny, 1] .- RHS[1:Nx, 1:Ny, 1])) < 1e-11
     @test std(result[1:Nx, 1:Ny, 1] .- RHS[1:Nx, 1:Ny, 1]) < 1e-13
    end


    return CUDA.@allowscalar result[1:Nx, 1:Ny, 1] ≈ RHS[1:Nx, 1:Ny, 1]
end

@testset "Implicit free surface solver tests" begin
    for arch in archs
        @info "Testing implicit free surface solver [$(typeof(arch))]..."
        # A spherical domain with bounded toplogy
        Nx = 360
        Ny = 360
        grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                    longitude = (-30, 30),
                                    latitude = (15, 75),
                                    z = (-4000, 0))
        @test run_implicit_free_surface_solver_tests(arch, grid)
        grid = RegularRectilinearGrid(size = (128, 1, 1),
                            x = (0, 1000kilometers), y = (0, 1), z = (-400, 0),
                            topology = (Bounded, Periodic, Bounded))
        @test run_implicit_free_surface_solver_tests(arch, grid)

    end
end
