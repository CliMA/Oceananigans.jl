using MPI
MPI.Init()
using Pkg

rank = MPI.Comm_rank(MPI.COMM_WORLD)

# instantiate only on rank 0
# to avoid problems with the filesystem
if rank == 0
    Pkg.develop(PackageSpec(path=pwd()))
    Pkg.instantiate()
end

MPI.Barrier(MPI.COMM_WORLD)
MPI.Finalize()