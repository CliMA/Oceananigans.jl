using Statistics
using Oceananigans.Buoyancy: g_Earth
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface, FreeSurface

function run_implicit_free_surface_solver_tests(arch)
    Nx = 360
    Ny = 360
    # A spherical domain with bounded toplogy
    grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                    longitude = (-30, 30),
                                    latitude = (15, 75),
                                    z = (-4000, 0)) 

    free_surface = ImplicitFreeSurface(gravitational_acceleration=g_Earth)

    # Create fields with default bounded and zeroflux boundaries.
    velocities = VelocityFields(arch, grid)
    RHS        = CenterField(arch, grid)

    # Initialize the solver
    free_surface = FreeSurface(free_surface, velocities, arch, grid)

    # Create a divergent velocity
    u, v, w = velocities
    imid = Int(floor(grid.Nx / 2)) + 1
    jmid = Int(floor(grid.Ny / 2)) + 1
    CUDA.@allowscalar u.data[imid, jmid, 1] = 1

    ### fill_halo_regions!(u.data, u.boundary_conditions, arch, grid)

    @test true

    return nothing
end

@testset "Implicit free surface solver tests" begin
    for arch in archs
        @info "Testing implicit free surface solver [$(typeof(arch))]..."
        run_implicit_free_surface_solver_tests(arch)
    end
end
