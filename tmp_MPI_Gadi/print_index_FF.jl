include("print_index_common.jl")

loc = (Face(), Face(), Center())
loc_name = "FF"

for fold_topology in [RightCenterFolded, RightFaceFolded]
    for partition in [Partition(1, 4), Partition(2, 2)]
        run_diagnostic(; loc, loc_name, fold_topology, partition,
                        global_Nx=10, global_Ny=10, Hx=2, Hy=2, Hz=2)
    end
end

MPI.Barrier(MPI.COMM_WORLD)
MPI.Finalize()
