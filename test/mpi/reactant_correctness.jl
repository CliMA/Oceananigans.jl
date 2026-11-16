include(joinpath(@__DIR__, "..", "setup", "reactant_test_utils.jl"))
include(joinpath(@__DIR__, "..", "setup", "reactant_correctness_utils.jl"))

using CUDA

all_combos(xs...) = vec(collect(Iterators.product(xs...)))

@testset "Reactant MPI correctness" begin
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    all_locations = all_combos((Center, Face), (Center, Face), (Center, Face))
    partitions = (Partition(4), Partition(1, 4), Partition(2, 2))

    rectilinear_grid(arch, topology) = RectilinearGrid(arch; size = (16, 16, 4), extent = (1, 1, 1), halo = (3, 3, 3), topology)

    function latitude_longitude_grid(arch, topology)
        longitude = topology[1] == Periodic ? (0, 360) : (0, 90)
        return LatitudeLongitudeGrid(arch; size = (16, 16, 4), halo = (3, 3, 3), longitude, latitude = (0, 10), z = (0, 1), topology)
    end

    rectilinear_topologies = ((Periodic, Periodic, Bounded), (Periodic, Bounded, Bounded), (Bounded, Bounded, Bounded))
    latitude_longitude_topologies = ((Periodic, Bounded, Bounded), (Bounded, Bounded, Bounded))

    # (grid constructor, topologies, Coriolis options)
    grids = ((rectilinear_grid, rectilinear_topologies, (nothing, FPlane(f=1e-4))),
             (latitude_longitude_grid, latitude_longitude_topologies, (nothing, HydrostaticSphericalCoriolis())))

    @testset "fill_halo_regions! on $make_grid with $partition, topology=$topology" for (make_grid, grid_topologies, _) in grids,
                                                                                         partition in partitions,
                                                                                         topology in grid_topologies
        vanilla_grid = make_grid(Distributed(child_arch; partition), topology)
        reactant_grid = make_grid(Distributed(ReactantState(); partition, communicator = MPI.COMM_WORLD), topology)

        for (LX, LY, LZ) in all_locations
            vanilla_field = Field{LX, LY, LZ}(vanilla_grid)
            reactant_field = Field{LX, LY, LZ}(reactant_grid)

            Random.seed!(12345 + rank)
            data = randn(size(vanilla_field)...)
            set!(vanilla_field, data)
            set!(reactant_field, data)

            fill_halo_regions!(vanilla_field)
            rfill! = @compile raise=true raise_first=true fill_halo_regions!(reactant_field)
            # rfill! = @compile fill_halo_regions!(reactant_field)
            rfill!(reactant_field)
            @test compare_parent("loc=$((LX, LY, LZ))", vanilla_field, reactant_field)
        end
    end

    advection_schemes = (nothing, Centered(), WENO())

    @testset "compute_simple_Gu! on $make_grid with $partition, topology=$topology" for (make_grid, grid_topologies, coriolis_options) in grids,
                                                                                          partition in partitions,
                                                                                          topology in grid_topologies
        vanilla_grid = make_grid(Distributed(child_arch; partition), topology)
        reactant_grid = make_grid(Distributed(ReactantState(); partition, communicator = MPI.COMM_WORLD), topology)

        vanilla_velocities = (u = XFaceField(vanilla_grid), v = YFaceField(vanilla_grid), w = ZFaceField(vanilla_grid))
        reactant_velocities = (u = XFaceField(reactant_grid), v = YFaceField(reactant_grid), w = ZFaceField(reactant_grid))

        Random.seed!(54321 + rank)
        for (vanilla_velocity, reactant_velocity) in zip(vanilla_velocities, reactant_velocities)
            data = randn(size(vanilla_velocity)...)
            set!(vanilla_velocity, data)
            set!(reactant_velocity, data)
            fill_halo_regions!(vanilla_velocity)
            rfill! = @compile raise=true raise_first=true fill_halo_regions!(reactant_velocity)
            rfill!(reactant_velocity)
        end

        vanilla_Gu = XFaceField(vanilla_grid)
        reactant_Gu = XFaceField(reactant_grid)

        for coriolis in coriolis_options, advection in advection_schemes
            advection = Oceananigans.Advection.materialize_advection(advection, reactant_grid)
            fill!(vanilla_Gu, 0)
            fill!(reactant_Gu, 0)
            compute_simple_Gu!(vanilla_Gu, advection, coriolis, vanilla_velocities)

            # raise=true is broken with WENO and a Bounded x-axis
            rcompute_Gu! = if advection isa WENO && topology[1] == Bounded
                @compile compute_simple_Gu!(reactant_Gu, advection, coriolis, reactant_velocities)
            else
                @compile raise=true raise_first=true compute_simple_Gu!(reactant_Gu, advection, coriolis, reactant_velocities)
            end
            rcompute_Gu!(reactant_Gu, advection, coriolis, reactant_velocities)
            @test compare_interior("Gu, advection=$(nameof(typeof(advection))), coriolis=$(nameof(typeof(coriolis)))",
                                   vanilla_Gu, reactant_Gu)
        end
    end
end
