# We need to initiate MPI for sharding because we are using a multi-host implementation: 
# i.e. we are launching the tests with `mpiexec` and on Github actions the default MPI 
# implementation is MPICH which requires calling MPI.Init(). In the case of OpenMPI,
# MPI.Init() is not necessary.
using MPI 
MPI.Init()
include("distributed_tests_utils.jl")

if Base.ARGS[1] == "tripolar"
    run_function = run_distributed_tripolar_grid
    suffix = "trg"
else
    run_function = run_distributed_latitude_longitude_grid
    suffix = "llg"
end

Reactant.Distributed.initialize(; single_gpu_per_process=false)

arch = Distributed(ReactantState(), partition = Partition(4, 1))
filename = "distributed_xslab_$(suffix).jld2"
run_function(arch, filename)

arch = Distributed(ReactantState(), partition = Partition(1, 4))
filename = "distributed_yslab_$(suffix).jld2"
run_function(arch, filename)

arch = Distributed(ReactantState(), partition = Partition(2, 2))
filename = "distributed_pencil_$(suffix).jld2"
run_function(arch, filename)
