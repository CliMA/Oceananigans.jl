include(joinpath(@__DIR__, "..", "setup", "reactant_test_utils.jl"))
include(joinpath(@__DIR__, "..", "setup", "reactant_correctness_utils.jl"))
include(joinpath(@__DIR__, "..", "setup", "reactant_mpi_correctness_utils.jl"))

using CUDA

@testset "Reactant MPI correctness" begin
    partitions = (Partition(4), Partition(1, 4), Partition(2, 2))
    rectilinear_topologies = ((Periodic, Periodic, Bounded), (Periodic, Bounded, Bounded), (Bounded, Bounded, Bounded))
    latitude_longitude_topologies = ((Periodic, Bounded, Bounded), (Bounded, Bounded, Bounded))

    # (grid constructor, topologies, Coriolis)
    grids = ((rectilinear_grid, rectilinear_topologies, FPlane(f=1e-4)),
             (latitude_longitude_grid, latitude_longitude_topologies, HydrostaticSphericalCoriolis()))

    @testset "fill_halo_regions! on $make_grid with $partition, topology=$topology" for (make_grid, grid_topologies, _) in grids,
                                                                                         partition in partitions,
                                                                                         topology in grid_topologies
        test_mpi_fill_halo_regions(make_grid, partition, topology)
    end

    @testset "compute_simple_Gu! on $make_grid, topology=$topology" for (make_grid, grid_topologies, coriolis) in grids,
                                                                        topology in grid_topologies
        test_mpi_compute_simple_Gu(make_grid, topology, coriolis)
    end
end
