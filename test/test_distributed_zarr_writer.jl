include("dependencies_for_runtests.jl")

using MPI
using Zarr

#####
##### ZarrWriter under MPI: driver test that launches mpiexec and verifies
##### the resulting Zarr store via a serial reader.
#####

const ZARR_MPI_WORKER = abspath(joinpath(@__DIR__, "distributed_zarr_writer_tests.jl"))

function run_zarr_mpi(nranks::Int, partition::String, out_path::String)
    isdir(out_path) && rm(out_path; recursive=true, force=true)
    log = read(`$(MPI.mpiexec()) -n $nranks $(Base.julia_cmd()) -O0 --project=$(Base.active_project()) $ZARR_MPI_WORKER --partition $partition --output $out_path`,
               String)
    return log
end

@testset "ZarrWriter [MPI]" begin
    @info "  Testing ZarrWriter under MPI (mpiexec-driven)..."

    @testset "Partition(x=2)" begin
        path = abspath(joinpath(@__DIR__, "test_dist_zarr_x.zarr"))
        log = run_zarr_mpi(2, "x", path)
        @test occursin("DISTRIBUTED_ZARR_OK", log)

        g = Zarr.zopen(path)
        @test "time" in keys(g.arrays)
        @test "u" in keys(g.arrays)
        times = g["time"][:]
        @test length(times) == 3
        @test times ≈ [0.0, 1.0, 2.0]

        # Global shape: (Nx_total, Ny_total, Nz, Nt) = (8, 8, 4, 3)
        u_arr = g["u"]
        @test size(u_arr) == (8, 8, 4, 3)

        # Rank topology recorded
        @test g.attrs["rank_topology"] == [2, 1, 1]

        # FieldTimeSeries reads the full global field
        u_fts = FieldTimeSeries(path, "u")
        @test length(u_fts.times) == 3
        @test size(u_fts.grid) == (8, 8, 4)

        rm(path; recursive=true, force=true)
    end

    @testset "Partition(y=2)" begin
        path = abspath(joinpath(@__DIR__, "test_dist_zarr_y.zarr"))
        log = run_zarr_mpi(2, "y", path)
        @test occursin("DISTRIBUTED_ZARR_OK", log)

        g = Zarr.zopen(path)
        @test size(g["u"]) == (8, 8, 4, 3)
        @test g.attrs["rank_topology"] == [1, 2, 1]
        rm(path; recursive=true, force=true)
    end

    @testset "Partition(x=2, y=2)" begin
        path = abspath(joinpath(@__DIR__, "test_dist_zarr_xy.zarr"))
        log = run_zarr_mpi(4, "xy", path)
        @test occursin("DISTRIBUTED_ZARR_OK", log)

        g = Zarr.zopen(path)
        @test size(g["u"]) == (8, 8, 4, 3)
        @test g.attrs["rank_topology"] == [2, 2, 1]
        rm(path; recursive=true, force=true)
    end
end
