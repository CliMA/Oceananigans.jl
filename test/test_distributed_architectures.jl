include("dependencies_for_runtests.jl")

using MPI
using Oceananigans.DistributedComputations
using CUDA

@testset "Distributed macros" begin
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    @onrank 0 begin
        @test rank == 0
    end

    @root begin
        @test rank == 0
    end

    @onrank 1 begin
        @test rank == 1
    end

    @onrank 2 begin
        @test rank == 2
    end

    @onrank 3 begin
        @test rank == 3
    end

    a = Int[]

    @distribute for i in 1:10
        push!(a, i)
    end

    @root begin
        @test a == [1, 5, 9]
    end

    @onrank 1 begin
        @test a == [2, 6, 10]
    end

    @onrank 2 begin
        @test a == [3, 7]
    end

    @onrank 3 begin
        @test a == [4, 8]
    end

    split_comm = MPI.Comm_split(MPI.COMM_WORLD, rank % 2, rank)

    a = Int[]

    @distribute split_comm for i in 1:10
        push!(a, i)
    end

    @onrank split_comm 0 @test a == [1, 3, 5, 7, 9]
    @onrank split_comm 1 @test a == [2, 4, 6, 8, 10]
end

@testset "Distributed architectures" begin
    for arch in test_architectures()
        child_arch = child_architecture(arch)
        if child_arch isa Oceananigans.Architectures.GPU
            # Check that no device is the same!
            local_comm = MPI.Comm_split_type(communicator, MPI.COMM_TYPE_SHARED, arch.local_rank)
            node_rank  = MPI.Comm_rank(local_comm)
            device_number = CUDA.device().handle
            # We are testing on the same node, therefore we can 
            # assume the GPU number changes with the rank
            @test node_rank == device_number
        end
    end
end
