include("reactant_test_utils.jl")
include("reactant_correctness_utils.jl")

using Random
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid

# Helper to generate all combinations
all_combos(xs...) = vec(collect(Iterators.product(xs...)))

@testset "Reactant correctness" begin
    @info "Testing Reactant correctness..."

    # Get vanilla architecture from TEST_ARCHITECTURE env var (set by reactant_test_utils.jl)
    vanilla_arch = get(ENV, "TEST_ARCHITECTURE", "CPU") == "GPU" ? GPU() : CPU()
    reactant_arch = ReactantState()

    # Field locations to test
    all_locations = all_combos((Center, Face), (Center, Face), (Center, Face))

    # Topologies to test for RectilinearGrid
    all_topologies = all_combos((Periodic, Bounded), (Periodic, Bounded), (Periodic, Bounded))

    #####
    ##### Halo filling tests
    #####

    @testset "fill_halo_regions! correctness" begin
        @info "  Testing fill_halo_regions!..."

        # Test RectilinearGrid with all topologies
        @testset "RectilinearGrid" begin
            @info "    RectilinearGrid..."
            for topo in all_topologies
                vanilla_grid = RectilinearGrid(vanilla_arch; size=(3, 4, 2), halo=(1, 1, 1),
                                               extent=(1, 1, 1), topology=topo)
                reactant_grid = RectilinearGrid(reactant_arch; size=(3, 4, 2), halo=(1, 1, 1),
                                                extent=(1, 1, 1), topology=topo)

                for loc in all_locations
                    LX, LY, LZ = loc
                    @testset "topo=$topo, loc=$loc" begin
                        vanilla_field = Field{LX, LY, LZ}(vanilla_grid)
                        reactant_field = Field{LX, LY, LZ}(reactant_grid)

                        Random.seed!(12345)
                        data = randn(size(vanilla_field)...)
                        set!(vanilla_field, data)
                        set!(reactant_field, data)

                        fill_halo_regions!(vanilla_field)
                        @jit fill_halo_regions!(reactant_field)

                        @test compare_parent("halo", vanilla_field, reactant_field)
                    end
                end
            end
        end

        # Test LatitudeLongitudeGrid
        @testset "LatitudeLongitudeGrid" begin
            @info "    LatitudeLongitudeGrid..."
            vanilla_grid = LatitudeLongitudeGrid(vanilla_arch; size=(4, 4, 2), halo=(1, 1, 1),
                                                 longitude=(0, 10), latitude=(0, 10), z=(0, 1))
            reactant_grid = LatitudeLongitudeGrid(reactant_arch; size=(4, 4, 2), halo=(1, 1, 1),
                                                  longitude=(0, 10), latitude=(0, 10), z=(0, 1))

            for loc in all_locations
                LX, LY, LZ = loc
                @testset "loc=$loc" begin
                    vanilla_field = Field{LX, LY, LZ}(vanilla_grid)
                    reactant_field = Field{LX, LY, LZ}(reactant_grid)

                    Random.seed!(12345)
                    data = randn(size(vanilla_field)...)
                    set!(vanilla_field, data)
                    set!(reactant_field, data)

                    fill_halo_regions!(vanilla_field)
                    @jit fill_halo_regions!(reactant_field)

                    @test compare_parent("halo", vanilla_field, reactant_field)
                end
            end
        end

        # Test TripolarGrid
        @testset "TripolarGrid" begin
            @info "    TripolarGrid..."
            vanilla_grid = TripolarGrid(vanilla_arch; size=(4, 4, 2), halo=(1, 1, 1),
                                        z=(0, 1), southernmost_latitude=-80)
            reactant_grid = TripolarGrid(reactant_arch; size=(4, 4, 2), halo=(1, 1, 1),
                                         z=(0, 1), southernmost_latitude=-80)

            for loc in all_locations
                LX, LY, LZ = loc
                @testset "loc=$loc" begin
                    vanilla_field = Field{LX, LY, LZ}(vanilla_grid)
                    reactant_field = Field{LX, LY, LZ}(reactant_grid)

                    Random.seed!(12345)
                    data = randn(size(vanilla_field)...)
                    set!(vanilla_field, data)
                    set!(reactant_field, data)

                    fill_halo_regions!(vanilla_field)
                    @jit fill_halo_regions!(reactant_field)

                    @test compare_parent("halo", vanilla_field, reactant_field)
                end
            end
        end
    end

    #####
    ##### Tendency computation tests
    #####

    @testset "compute_simple_Gu! correctness" begin
        @info "  Testing compute_simple_Gu!..."

        advection_schemes = (Centered(), WENO())

        # RectilinearGrid + FPlane
        @testset "RectilinearGrid + FPlane" begin
            @info "    RectilinearGrid + FPlane..."
            coriolis = FPlane(f=1e-4)

            vanilla_grid = RectilinearGrid(vanilla_arch; size=(4, 4, 4), halo=(3, 3, 3),
                                           extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
            reactant_grid = RectilinearGrid(reactant_arch; size=(4, 4, 4), halo=(3, 3, 3),
                                            extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))

            # Set up velocity fields
            vanilla_velocities = (u=XFaceField(vanilla_grid), v=YFaceField(vanilla_grid), w=ZFaceField(vanilla_grid))
            reactant_velocities = (u=XFaceField(reactant_grid), v=YFaceField(reactant_grid), w=ZFaceField(reactant_grid))

            Random.seed!(54321)
            for (vf, rf) in zip(vanilla_velocities, reactant_velocities)
                data = randn(size(vf)...)
                set!(vf, data)
                set!(rf, data)
                fill_halo_regions!(vf)
                @jit fill_halo_regions!(rf)
            end

            vanilla_Gu = XFaceField(vanilla_grid)
            reactant_Gu = XFaceField(reactant_grid)

            for advection in advection_schemes
                @testset "advection=$(nameof(typeof(advection)))" begin
                    fill!(vanilla_Gu, 0)
                    fill!(reactant_Gu, 0)

                    compute_simple_Gu!(vanilla_Gu, advection, coriolis, vanilla_velocities)
                    @jit compute_simple_Gu!(reactant_Gu, advection, coriolis, reactant_velocities)

                    @test compare_parent("Gu", vanilla_Gu, reactant_Gu)
                end
            end
        end

        # LatitudeLongitudeGrid + HydrostaticSphericalCoriolis
        @testset "LatitudeLongitudeGrid + HydrostaticSphericalCoriolis" begin
            @info "    LatitudeLongitudeGrid + HydrostaticSphericalCoriolis..."
            coriolis = HydrostaticSphericalCoriolis()

            vanilla_grid = LatitudeLongitudeGrid(vanilla_arch; size=(4, 4, 4), halo=(3, 3, 3),
                                                 longitude=(0, 10), latitude=(0, 10), z=(0, 1))
            reactant_grid = LatitudeLongitudeGrid(reactant_arch; size=(4, 4, 4), halo=(3, 3, 3),
                                                  longitude=(0, 10), latitude=(0, 10), z=(0, 1))

            # Set up velocity fields
            vanilla_velocities = (u=XFaceField(vanilla_grid), v=YFaceField(vanilla_grid), w=ZFaceField(vanilla_grid))
            reactant_velocities = (u=XFaceField(reactant_grid), v=YFaceField(reactant_grid), w=ZFaceField(reactant_grid))

            Random.seed!(54321)
            for (vf, rf) in zip(vanilla_velocities, reactant_velocities)
                data = randn(size(vf)...)
                set!(vf, data)
                set!(rf, data)
                fill_halo_regions!(vf)
                @jit fill_halo_regions!(rf)
            end

            vanilla_Gu = XFaceField(vanilla_grid)
            reactant_Gu = XFaceField(reactant_grid)

            for advection in advection_schemes
                @testset "advection=$(nameof(typeof(advection)))" begin
                    fill!(vanilla_Gu, 0)
                    fill!(reactant_Gu, 0)

                    compute_simple_Gu!(vanilla_Gu, advection, coriolis, vanilla_velocities)
                    @jit compute_simple_Gu!(reactant_Gu, advection, coriolis, reactant_velocities)

                    @test compare_parent("Gu", vanilla_Gu, reactant_Gu)
                end
            end
        end
    end
end

# TODO: Add tests with @jit raise=true for autodiff compatibility
# See https://github.com/CliMA/Oceananigans.jl/pull/5093 for discussion
