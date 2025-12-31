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
end
