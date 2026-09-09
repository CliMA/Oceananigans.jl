include("dependencies_for_runtests.jl")

isdefined(Main, :run_cubed_sphere_simulation_test) ||
    include("test_multi_region_cubed_sphere_simulation_utils.jl")

@testset "Testing simulation on immersed conformal cubed sphere grids" begin
    for non_uniform_conformal_mapping in (false, true)
        cm = non_uniform_conformal_mapping ? "non-uniform conformal mapping" : "uniform conformal mapping"
        cm_suffix = non_uniform_conformal_mapping ? "NUCM" : "UCM"
        for FT in (Oceananigans.defaults.FloatType,), arch in archs
            Nx, Ny, Nz = 32, 32, 10

            underlying_grid = ConformalCubedSphereGrid(arch, FT;
                                                       panel_size = (Nx, Ny, Nz), z = (-3000, 0),
                                                       radius = Oceananigans.defaults.planet_radius,
                                                       horizontal_direction_halo = 6, non_uniform_conformal_mapping)

            @inline bottom(λ, φ) = ifelse(abs(φ) < 30, - 2, 0)
            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom);
                                        active_cells_map = true)

            run_cubed_sphere_simulation_test(grid, "IG", FT, arch, cm, cm_suffix)
        end
    end
end
