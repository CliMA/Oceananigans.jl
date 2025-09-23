    using MPI
    MPI.Init()
    using Oceananigans.DistributedComputations: Equal

    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(y = Equal()))
    run_distributed_tripolar_grid(arch, "distributed_yslab_tripolar.jld2")
