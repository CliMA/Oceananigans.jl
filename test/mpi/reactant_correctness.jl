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

    @testset "fill_halo_regions! on TripolarGrid with $partition, fold=$fold" for partition in (Partition(1, 4), Partition(2, 2)),
                                                                                  fold in ((CenterFolded, UPivot), (CenterFolded, TPivot), (FaceFolded, FPivot))
        test_mpi_fill_halo_regions(tripolar_grid, partition, fold)
    end

    @testset "untraced fill_halo_regions!" begin
        test_mpi_fill_halo_regions(rectilinear_grid, Partition(2, 2), (Periodic, Periodic, Bounded); traced=false)
    end

    @testset "compute_simple_Gu! on $make_grid, topology=$topology" for (make_grid, grid_topologies, coriolis) in grids,
                                                                        topology in grid_topologies
        test_mpi_compute_simple_Gu(make_grid, topology, coriolis)
    end

    # 32×32 gives 16×16 local grids, so the split-explicit extended halos fit without capping.
    # 16×16 gives 8×8 local grids, which caps the extended halos and adds extra barotropic halo fills.
    hydrostatic_cases = (((Periodic, Bounded, Bounded), 32, "uncapped extended halos"),
                         ((Bounded, Bounded, Bounded), 16, "capped extended halos"))

    @testset "HydrostaticFreeSurfaceModel time stepping with Partition(2, 2), topology=$topology, $description" for (topology, N, description) in hydrostatic_cases
        test_mpi_hydrostatic_free_surface_model(Partition(2, 2), topology; Nx=N, Ny=N)
    end
end
