using Test

include("reactant_test_utils.jl")
include("distributed_tripolar_tests_utils.jl")

# Here, we reuse the tests performed in `test_distributed_tripolar.jl`, to check that
# the sharding is performed correctly.

# We are running on 8 "fake" CPUs
ENV["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

@testset "Sharded tripolar grid and fields" begin
    child_arch = ReactantState()

    archs = [Distributed(child_arch, partition=Partition(1, 8)),
             Distributed(child_arch, partition=Partition(2, 4)),
             Distributed(child_arch, partition=Partition(4, 2))]

    for arch in archs
        @info "  Testing a tripolar grid on a $(arch.ranks) partition"
        local_grid  = TripolarGrid(arch; size = (40, 40, 1), z = (-1000, 0), halo = (2, 2, 2))
        global_grid = TripolarGrid(child_arch; size = (40, 40, 1), z = (-1000, 0), halo = (2, 2, 2))

        # Actually nothing for the moment...
    end
end
