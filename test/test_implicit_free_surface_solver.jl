using Statistics
using Oceananigans.Buoyancy: g_Earth
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface, FreeSurface, 
      compute_volume_scaled_divergence!, add_previous_free_surface_contribution

function run_implicit_free_surface_solver_tests(arch)
    Nx = 360
    Ny = 360
    Δt = 900
    # A spherical domain with bounded toplogy
    grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                    longitude = (-30, 30),
                                    latitude = (15, 75),
                                    z = (-4000, 0)) 

    # A boring grid
    ## RegularRectilinearGrid(FT, size=(3, 1, 4), extent=(3, 1, 4))

    free_surface = ImplicitFreeSurface(gravitational_acceleration=g_Earth)

    # Create fields with default bounded and zeroflux boundaries.
    velocities = VelocityFields(arch, grid)
    η          = CenterField(arch, grid)
    @. η.data  = 0

    # Initialize the solver
    free_surface = FreeSurface(free_surface, velocities, arch, grid)

    # Create a divergent velocity
    u, v, w = velocities
    imid = Int(floor(grid.Nx / 2)) + 1
    jmid = Int(floor(grid.Ny / 2)) + 1
    CUDA.@allowscalar u.data[imid, jmid, 1] = 1

    fill_halo_regions!(u.data, u.boundary_conditions, arch, grid, nothing, nothing)

    ### event = launch!(arch, grid, :xyz, divergence!, grid, u.data, v.data, w.data, RHS.data,
    ###                 dependencies=Event(device(arch)))
    ### wait(device(arch), event)

    # Create a fake model
    model = (architecture=arch,grid=grid,free_surface=free_surface)

    # Calculate the vertically ingrated divergence term
    event = compute_volume_scaled_divergence!(free_surface, model)
    wait(device(model.architecture), event)

    # Scale and add in η term
    RHS = free_surface.implicit_step_solver.solver.settings.RHS
    RHS .= RHS/(free_surface.gravitational_acceleration*Δt)
    event = add_previous_free_surface_contribution(free_surface, model, Δt )
    wait(device(model.architecture), event)
    fill_halo_regions!(RHS   , η.boundary_conditions, model.architecture, model.grid)


    @test true

    return nothing
end

@testset "Implicit free surface solver tests" begin
    for arch in archs
        @info "Testing implicit free surface solver [$(typeof(arch))]..."
        run_implicit_free_surface_solver_tests(arch)
    end
end
