include(joinpath(@__DIR__, "..", "setup", "reactant_test_utils.jl"))
include(joinpath(@__DIR__, "..", "setup", "reactant_correctness_utils.jl"))

using CUDA

all_combos(xs...) = vec(collect(Iterators.product(xs...)))

@testset "Reactant MPI correctness" begin
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    all_locations = all_combos((Center, Face), (Center, Face), (Center, Face))
    topologies = ((Periodic, Periodic, Bounded), (Periodic, Bounded, Bounded), (Bounded, Bounded, Bounded))
    partitions = (Partition(4), Partition(1, 4), Partition(2, 2))

    @testset "fill_halo_regions! with $partition, topology=$topology" for partition in partitions, topology in topologies
        kw = (; size = (8, 8, 4), extent = (1, 1, 1), halo = (3, 3, 3), topology)
        vanilla_grid = RectilinearGrid(Distributed(child_arch; partition); kw...)
        reactant_grid = RectilinearGrid(Distributed(ReactantState(); partition, communicator = MPI.COMM_WORLD); kw...)

        for (LX, LY, LZ) in all_locations
            vanilla_field = Field{LX, LY, LZ}(vanilla_grid)
            reactant_field = Field{LX, LY, LZ}(reactant_grid)

            Random.seed!(12345 + rank)
            data = randn(size(vanilla_field)...)
            set!(vanilla_field, data)
            set!(reactant_field, data)

            fill_halo_regions!(vanilla_field)
            # fails for some cases
            rfill! = @compile raise=true raise_first=true fill_halo_regions!(reactant_field)
            # # succeeds in all cases without raise=true
            # rfill! = @compile fill_halo_regions!(reactant_field)
            rfill!(reactant_field)
            @test compare_parent("loc=$((LX, LY, LZ))", vanilla_field, reactant_field)
        end
    end
end
