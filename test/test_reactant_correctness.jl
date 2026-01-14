include("reactant_test_utils.jl")
include("reactant_correctness_utils.jl")

using Random
using CUDA
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid

# Helper to generate all combinations
all_combos(xs...) = vec(collect(Iterators.product(xs...)))

@testset "Reactant correctness" begin
    @info "Testing Reactant correctness (comparing vanilla Oceananigans vs ReactantState)..."

    # Get vanilla architecture from TEST_ARCHITECTURE env var (set by reactant_test_utils.jl)
    vanilla_arch = get(ENV, "TEST_ARCHITECTURE", "CPU") == "GPU" ? GPU() : CPU()
    reactant_arch = ReactantState()

    # Field locations to test
    all_locations = all_combos((Center, Face), (Center, Face), (Center, Face))

    # Topologies to test for RectilinearGrid
    # Note: We exclude (Periodic, Periodic, Periodic) because it triggers a segfault in
    # Reactant's MLIR pattern rewriting when using raise=true (RecognizeRotate pass bug)
    all_topologies = filter(topo -> topo != (Periodic, Periodic, Periodic),
                            all_combos((Periodic, Bounded), (Periodic, Bounded), (Periodic, Bounded)))

    # JIT raise modes: raise=false is default, raise=true is needed for autodiff
    raise_modes = (false, true)

    #####
    ##### Halo filling tests
    #####

    @testset "fill_halo_regions! correctness" begin

        # Test RectilinearGrid with all topologies
        for topo in all_topologies
            @testset "RectilinearGrid topology=$topo" begin
                @info "Testing fill_halo_regions! correctness on RectilinearGrid with topology=$topo..."
                vanilla_grid = RectilinearGrid(vanilla_arch; size=(3, 4, 2), halo=(1, 1, 1),
                                               extent=(1, 1, 1), topology=topo)
                reactant_grid = RectilinearGrid(reactant_arch; size=(3, 4, 2), halo=(1, 1, 1),
                                                extent=(1, 1, 1), topology=topo)

                for loc in all_locations, raise in raise_modes
                    LX, LY, LZ = loc
                    @testset "loc=$loc raise=$raise" begin
                        vanilla_field = Field{LX, LY, LZ}(vanilla_grid)
                        reactant_field = Field{LX, LY, LZ}(reactant_grid)

                        Random.seed!(12345)
                        data = randn(size(vanilla_field)...)
                        set!(vanilla_field, data)
                        set!(reactant_field, data)

                        fill_halo_regions!(vanilla_field)
                        @jit raise=raise fill_halo_regions!(reactant_field)

                        @test compare_parent("halo", vanilla_field, reactant_field)
                    end
                end
            end
        end

        # Test LatitudeLongitudeGrid with LX ∈ (Periodic, Bounded)
        for LX in (Periodic, Bounded)
            topo = (LX, Bounded, Bounded)
            @testset "LatitudeLongitudeGrid topology=$topo" begin
                @info "Testing fill_halo_regions! correctness on LatitudeLongitudeGrid with topology=$topo..."
                vanilla_grid = LatitudeLongitudeGrid(vanilla_arch; size=(4, 4, 2), halo=(1, 1, 1),
                                                     longitude=(0, 10), latitude=(0, 10), z=(0, 1),
                                                     topology=topo)
                reactant_grid = LatitudeLongitudeGrid(reactant_arch; size=(4, 4, 2), halo=(1, 1, 1),
                                                      longitude=(0, 10), latitude=(0, 10), z=(0, 1),
                                                      topology=topo)

                for loc in all_locations, raise in raise_modes
                    LXf, LYf, LZf = loc
                    @testset "loc=$loc raise=$raise" begin
                        vanilla_field = Field{LXf, LYf, LZf}(vanilla_grid)
                        reactant_field = Field{LXf, LYf, LZf}(reactant_grid)

                        Random.seed!(12345)
                        data = randn(size(vanilla_field)...)
                        set!(vanilla_field, data)
                        set!(reactant_field, data)

                        fill_halo_regions!(vanilla_field)
                        @jit raise=raise fill_halo_regions!(reactant_field)

                        @test compare_parent("halo", vanilla_field, reactant_field)
                    end
                end
            end
        end

        # Test TripolarGrid (fixed topology)
        @testset "TripolarGrid" begin
            @info "Testing fill_halo_regions! correctness on TripolarGrid..."
            vanilla_grid = TripolarGrid(vanilla_arch; size=(4, 4, 2), halo=(1, 1, 1),
                                        z=(0, 1), southernmost_latitude=-80)
            reactant_grid = TripolarGrid(reactant_arch; size=(4, 4, 2), halo=(1, 1, 1),
                                         z=(0, 1), southernmost_latitude=-80)

            for loc in all_locations, raise in raise_modes
                LX, LY, LZ = loc
                @testset "loc=$loc raise=$raise" begin
                    vanilla_field = Field{LX, LY, LZ}(vanilla_grid)
                    reactant_field = Field{LX, LY, LZ}(reactant_grid)

                    Random.seed!(12345)
                    data = randn(size(vanilla_field)...)
                    set!(vanilla_field, data)
                    set!(reactant_field, data)

                    fill_halo_regions!(vanilla_field)
                    @jit raise=raise fill_halo_regions!(reactant_field)

                    @test compare_parent("halo", vanilla_field, reactant_field)
                end
            end
        end
    end

    #####
    ##### Tendency computation tests
    #####

    @testset "compute_simple_Gu! correctness" begin

        advection_schemes = (Centered(), WENO())

        # RectilinearGrid + FPlane
        @testset "RectilinearGrid + FPlane" begin
            @info "Testing compute_simple_Gu! correctness on RectilinearGrid with FPlane Coriolis..."
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

            for advection in advection_schemes, raise in raise_modes
                adv_name = nameof(typeof(advection))
                @testset "advection=$adv_name raise=$raise" begin
                    fill!(vanilla_Gu, 0)
                    fill!(reactant_Gu, 0)

                    compute_simple_Gu!(vanilla_Gu, advection, coriolis, vanilla_velocities)
                    @jit raise=raise compute_simple_Gu!(reactant_Gu, advection, coriolis, reactant_velocities)

                    @test compare_parent("Gu", vanilla_Gu, reactant_Gu)
                end
            end
        end

        # LatitudeLongitudeGrid + HydrostaticSphericalCoriolis
        @testset "LatitudeLongitudeGrid + HydrostaticSphericalCoriolis" begin
            @info "Testing compute_simple_Gu! correctness on LatitudeLongitudeGrid with HydrostaticSphericalCoriolis..."
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

            for advection in advection_schemes, raise in raise_modes
                adv_name = nameof(typeof(advection))
                @testset "advection=$adv_name raise=$raise" begin
                    fill!(vanilla_Gu, 0)
                    fill!(reactant_Gu, 0)

                    compute_simple_Gu!(vanilla_Gu, advection, coriolis, vanilla_velocities)
                    @jit raise=raise compute_simple_Gu!(reactant_Gu, advection, coriolis, reactant_velocities)

                    @test compare_parent("Gu", vanilla_Gu, reactant_Gu)
                end
            end
        end
    end
end
