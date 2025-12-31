include("reactant_test_utils.jl")
include("reactant_correctness_utils.jl")

using Random
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid

@testset "Reactant correctness" begin
    @info "Testing Reactant correctness..."

    # Get vanilla architecture from TEST_ARCHITECTURE env var (set by reactant_test_utils.jl)
    vanilla_arch = get(ENV, "TEST_ARCHITECTURE", "CPU") == "GPU" ? GPU() : CPU()
    reactant_arch = ReactantState()

    # Locations to test: all combinations of Center and Face
    locations = vec(collect(Iterators.product(
        (Center, Face),
        (Center, Face),
        (Center, Face)
    )))

    # Grid configurations: each entry is (GridType, grid_kwargs, topologies_to_test, name)
    # We test different topologies for RectilinearGrid, but fixed topologies for spherical grids
    grid_configs = [
        # RectilinearGrid: test all topology combinations
        (
            RectilinearGrid,
            (; size=(3, 4, 2), halo=(1, 1, 1), extent=(1.0, 1.0, 1.0)),
            vec(collect(Iterators.product(
                (Periodic, Bounded),
                (Periodic, Bounded),
                (Periodic, Bounded)
            ))),
            "RectilinearGrid"
        ),
        # LatitudeLongitudeGrid: Periodic in longitude, Bounded in latitude and z
        (
            LatitudeLongitudeGrid,
            (; size=(4, 4, 2), halo=(1, 1, 1), longitude=(0, 10), latitude=(0, 10), z=(0, 1)),
            [(Periodic, Bounded, Bounded)],
            "LatitudeLongitudeGrid"
        ),
        # TripolarGrid: fixed topology (Periodic, RightConnected, Bounded)
        # Note: Nx must be even for TripolarGrid
        (
            TripolarGrid,
            (; size=(4, 4, 2), halo=(1, 1, 1), z=(0, 1), southernmost_latitude=-80),
            nothing,  # TripolarGrid has fixed topology, no need to specify
            "TripolarGrid"
        ),
    ]

    @testset "fill_halo_regions! matches vanilla" begin
        @info "  Testing fill_halo_regions! equivalence across grid types, topologies, and locations..."

        for (GridType, grid_kw, topologies, grid_name) in grid_configs
            @testset "$grid_name" begin
                @info "    Testing $grid_name..."

                # For grids with fixed topology (like TripolarGrid), just build the grid directly
                if isnothing(topologies)
                    # Build grids without specifying topology (use grid's default)
                    vanilla_grid = GridType(vanilla_arch; grid_kw...)
                    reactant_grid = GridType(reactant_arch; grid_kw...)

                    for loc in locations
                        LX, LY, LZ = loc
                        loc_name = "$(LX)×$(LY)×$(LZ)"

                        @testset "$loc_name" begin
                            # Build fields at the specified location
                            vanilla_field = Field{LX, LY, LZ}(vanilla_grid)
                            reactant_field = Field{LX, LY, LZ}(reactant_grid)

                            # Seed randomness deterministically
                            Random.seed!(12345)
                            init_data = randn(Float64, size(vanilla_field)...)

                            # Set fields with the same initial data
                            set!(vanilla_field, init_data)
                            set!(reactant_field, init_data)

                            # Fill halos on both
                            fill_halo_regions!(vanilla_field)
                            @jit fill_halo_regions!(reactant_field)

                            # Test: parent arrays (including halos) should match
                            @test compare_parent("halo", vanilla_field, reactant_field)
                        end
                    end
                else
                    # For grids with configurable topology, loop over topologies
                    for topo in topologies
                        TX, TY, TZ = topo
                        topo_name = "$(TX)×$(TY)×$(TZ)"

                        @testset "$topo_name" begin
                            # Build grids with specified topology
                            full_grid_kw = (; grid_kw..., topology=(TX, TY, TZ))
                            vanilla_grid = GridType(vanilla_arch; full_grid_kw...)
                            reactant_grid = GridType(reactant_arch; full_grid_kw...)

                            for loc in locations
                                LX, LY, LZ = loc
                                loc_name = "$(LX)×$(LY)×$(LZ)"

                                @testset "$loc_name" begin
                                    # Build fields at the specified location
                                    vanilla_field = Field{LX, LY, LZ}(vanilla_grid)
                                    reactant_field = Field{LX, LY, LZ}(reactant_grid)

                                    # Seed randomness deterministically
                                    Random.seed!(12345)
                                    init_data = randn(Float64, size(vanilla_field)...)

                                    # Set fields with the same initial data
                                    set!(vanilla_field, init_data)
                                    set!(reactant_field, init_data)

                                    # Fill halos on both
                                    fill_halo_regions!(vanilla_field)
                                    @jit fill_halo_regions!(reactant_field)

                                    # Test: parent arrays (including halos) should match
                                    @test compare_parent("halo", vanilla_field, reactant_field)
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    @testset "compute_simple_Gu! matches vanilla" begin
        @info "  Testing simple u-tendency (advection + Coriolis) equivalence..."

        # Advection schemes to test
        # Note: WENOVectorInvariant requires larger halos due to vorticity stencil, skipped for now
        advection_schemes = [
            (Centered(), "Centered"),
            (WENO(), "WENO"),
        ]

        # Grid + Coriolis configurations
        # Each entry: (GridType, grid_kwargs, coriolis, name)
        # Note: TripolarGrid requires special handling at north pole (Zipper BC) and produces NaN
        # with random data, so we skip it for this test. Halo filling tests for TripolarGrid still pass.
        Gu_grid_configs = [
            # RectilinearGrid with FPlane
            (
                RectilinearGrid,
                (; size=(4, 4, 4), halo=(3, 3, 3), extent=(1.0, 1.0, 1.0), topology=(Periodic, Periodic, Bounded)),
                FPlane(f=1e-4),
                "RectilinearGrid+FPlane"
            ),
            # LatitudeLongitudeGrid with HydrostaticSphericalCoriolis
            (
                LatitudeLongitudeGrid,
                (; size=(4, 4, 4), halo=(3, 3, 3), longitude=(0, 10), latitude=(0, 10), z=(0, 1)),
                HydrostaticSphericalCoriolis(),
                "LatLonGrid+HydrostaticSphericalCoriolis"
            ),
        ]

        for (GridType, grid_kw, coriolis, grid_name) in Gu_grid_configs
            @testset "$grid_name" begin
                @info "    Testing $grid_name..."

                vanilla_grid = GridType(vanilla_arch; grid_kw...)
                reactant_grid = GridType(reactant_arch; grid_kw...)

                # Create velocity fields for both architectures
                vanilla_u = XFaceField(vanilla_grid)
                vanilla_v = YFaceField(vanilla_grid)
                vanilla_w = ZFaceField(vanilla_grid)
                vanilla_velocities = (; u=vanilla_u, v=vanilla_v, w=vanilla_w)

                reactant_u = XFaceField(reactant_grid)
                reactant_v = YFaceField(reactant_grid)
                reactant_w = ZFaceField(reactant_grid)
                reactant_velocities = (; u=reactant_u, v=reactant_v, w=reactant_w)

                # Create Gu fields to store the tendency
                vanilla_Gu = XFaceField(vanilla_grid)
                reactant_Gu = XFaceField(reactant_grid)

                # Initialize velocity fields with random data
                Random.seed!(54321)
                u_data = randn(Float64, size(vanilla_u)...)
                v_data = randn(Float64, size(vanilla_v)...)
                w_data = randn(Float64, size(vanilla_w)...)

                set!(vanilla_u, u_data)
                set!(vanilla_v, v_data)
                set!(vanilla_w, w_data)

                set!(reactant_u, u_data)
                set!(reactant_v, v_data)
                set!(reactant_w, w_data)

                # Fill halos for velocities
                fill_halo_regions!(vanilla_u)
                fill_halo_regions!(vanilla_v)
                fill_halo_regions!(vanilla_w)

                @jit fill_halo_regions!(reactant_u)
                @jit fill_halo_regions!(reactant_v)
                @jit fill_halo_regions!(reactant_w)

                for (advection, adv_name) in advection_schemes
                    @testset "$adv_name" begin
                        # Reset Gu fields
                        fill!(vanilla_Gu, 0)
                        fill!(reactant_Gu, 0)

                        # Compute simplified Gu on both
                        compute_simple_Gu!(vanilla_Gu, advection, coriolis, vanilla_velocities)
                        @jit compute_simple_Gu!(reactant_Gu, advection, coriolis, reactant_velocities)

                        # Compare the tendency fields
                        @test compare_parent("Gu", vanilla_Gu, reactant_Gu)
                    end
                end
            end
        end
    end
end
