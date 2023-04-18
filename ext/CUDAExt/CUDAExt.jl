module CUDAExt
    using Oceananigans
    using CUDA

    function __init__()
        if CUDA.has_cuda()
            @debug "CUDA-enabled GPU(s) detected:"
            for (gpu, dev) in enumerate(CUDA.devices())
                @debug "$dev: $(CUDA.name(dev))"
            end

            CUDA.allowscalar(false)
        end
    end

    include("Architectures.jl") 
    include("BoundaryConditions.jl") 
end
