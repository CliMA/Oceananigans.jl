include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using MPI
using Zarr

MPI.Initialized() || MPI.Init(threadlevel=:multiple)

# Every rank's slab, including the fold rows, must land in the single store where a serial grid puts it.
@testset "ZarrWriter on a distributed TripolarGrid" begin
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    u₀(λ, φ, z) = 10λ + φ
    c₀(λ, φ, z) = 100λ + φ + z

    for fold_topology in (CenterFolded, FaceFolded), (Rx, Ry) in ((1, 4), (2, 2))
        @info "  Testing ZarrWriter on a $fold_topology TripolarGrid with Partition($Rx, $Ry) [$(typeof(child_arch))]..."
        arch = Distributed(child_arch; partition=Partition(Rx, Ry))
        kw = (size=(16, 12, 4), z=(-1, 0), fold_topology)

        model = HydrostaticFreeSurfaceModel(TripolarGrid(arch; kw...); tracers=:c)
        set!(model, u=u₀, c=c₀)

        path = abspath("distributed_tripolar_$(fold_topology)_$(Rx)x$(Ry).zarr")
        writer = ZarrWriter(model, (u=model.velocities.u, c=model.tracers.c);
                            filename=path, schedule=IterationInterval(1), overwrite_files=true)
        Oceananigans.initialize!(writer, model)
        Oceananigans.write_output!(writer, model)
        MPI.Barrier(MPI.COMM_WORLD)

        if rank == 0
            serial = HydrostaticFreeSurfaceModel(TripolarGrid(CPU(); kw...); tracers=:c)
            set!(serial, u=u₀, c=c₀)

            store = Zarr.zopen(path)
            @test store["u"][:, :, :, 1] == Float32.(interior(serial.velocities.u))
            @test store["c"][:, :, :, 1] == Float32.(interior(serial.tracers.c))
            @test size(FieldTimeSeries(path, "c").grid) == (16, 12, 4)

            rm(path; recursive=true, force=true)
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end
