using Statistics
using Oceananigans.BuoyancyModels: g_Earth

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    ImplicitFreeSurface,
    FreeSurface,
    implicit_free_surface_step!,
    implicit_free_surface_linear_operation!

function run_implicit_free_surface_solver_tests(arch, grid)

    Δt = 900
    Nx = grid.Nx
    Ny = grid.Ny

    # Create a model
    model = HydrostaticFreeSurfaceModel(architecture = arch,
                                        grid = grid,
                                        momentum_advection = nothing,
                                        free_surface=ImplicitFreeSurface())
    
    # Create a divergent velocity
    u, v, w = model.velocities
    imid = Int(floor(grid.Nx / 2)) + 1
    jmid = Int(floor(grid.Ny / 2)) + 1
    CUDA.@allowscalar u[imid, jmid, 1] = 1

    implicit_free_surface_step!(model.free_surface, model, Δt, 1.5, Event(device(arch)))

    # Extract right hand side "truth"
    right_hand_side = model.free_surface.implicit_step_right_hand_side

    # Compute left hand side "solution"
    g = g_Earth
    η = model.free_surface.η
    ∫ᶻ_Axᶠᶜᶜ = model.free_surface.vertically_integrated_lateral_face_areas.xᶠᶜᶜ
    ∫ᶻ_Ayᶜᶠᶜ = model.free_surface.vertically_integrated_lateral_face_areas.yᶜᶠᶜ

    left_hand_side = ReducedField(Center, Center, Nothing, arch, grid; dims=3)
    implicit_free_surface_linear_operation!(left_hand_side, η, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)

    # Compare
    extrema_tolerance = 1e-9
    std_tolerance = 1e-9

    CUDA.@allowscalar begin
        @test minimum(abs, interior(left_hand_side) .- interior(right_hand_side)) < extrema_tolerance
        @test maximum(abs, interior(left_hand_side) .- interior(right_hand_side)) < extrema_tolerance
        @test std(interior(left_hand_side) .- interior(right_hand_side)) < std_tolerance
    end

    return nothing
end

@testset "Implicit free surface solver tests" begin
    for arch in archs

        rectilinear_grid = RegularRectilinearGrid(size = (128, 1, 5),
                                                  x = (0, 1000kilometers), y = (0, 1), z = (-400, 0),
                                                  topology = (Bounded, Periodic, Bounded))

        lat_lon_grid = RegularLatitudeLongitudeGrid(size = (90, 90, 5),
                                                    longitude = (-30, 30),
                                                    latitude = (15, 75),
                                                    z = (-4000, 0))

        for grid in (rectilinear_grid, lat_lon_grid)
            @info "Testing implicit free surface solver [$(typeof(arch)), $(typeof(grid).name.wrapper)]..."
            run_implicit_free_surface_solver_tests(arch, grid)
        end
    end
end
