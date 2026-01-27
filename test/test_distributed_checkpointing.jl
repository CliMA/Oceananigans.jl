include("dependencies_for_runtests.jl")
include("dependencies_for_checkpoint_tests.jl")

using Oceananigans.DistributedComputations

arch = Distributed(CPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)

for model_type in (:nonhydrostatic, :hydrostatic)
    for pickup_method in (:boolean, :iteration, :filepath)
        @testset "Minimal distributed restore [$model_type, $pickup_method] [$(typeof(arch))]" begin
            @info "  Testing minimal distributed restore [$model_type, $pickup_method] [$(typeof(arch))]..."
            test_minimal_restore(arch, Float64, pickup_method, model_type)
        end
    end
end
